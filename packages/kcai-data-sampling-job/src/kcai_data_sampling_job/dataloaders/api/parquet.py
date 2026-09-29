"""The ``parquet`` image dataloader and its plugin-owned config.

Parquet is the only input. Each row is one sample and carries
exactly one processable image column (``sample_path.column``), holding one of
three forms: an **absolute path**, a **relative path** (+``sample_path.prefix``),
or **raw image bytes**. The loader normalizes each ``load_batch_size`` chunk to
a decoded batch = row: the non-image source columns stay a pyarrow table and
the image column becomes one numpy ``(B, H, W, 4)`` uint8 stack — zero-copy
``np.frombuffer`` for bytes columns, Pillow decode for path columns (the codec
from ``-job/utils/images.py``). Filters, casts and path-prefix handling port
from dqm-ml, with the batch-size rename carried over and the ``split`` branch
dropped (kcai has no gap concept): one selection per loader, named
``<loader>``.

The loader builds the generic core :class:`Batch` (``-job`` depends on core
only), fixing the image axes and value range inline. Its datatype-specific
config — :class:`ParquetImageLoaderConfig` — is the plugin's own registered
schema, resolved against the generic loader keys at config load.
"""

from collections.abc import Callable, Iterator
import logging
from pathlib import Path
from typing import Any, Literal

from kcai_data_sampling_core.api.dataloaders import DataLoader, DataSelection
from kcai_data_sampling_core.api.selection import Batch
from kcai_data_sampling_core.models.dataloaders import DataLoaderConfig
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import Field
from typing_extensions import override

from kcai_data_sampling_job.utils.images import decode_image

logger = logging.getLogger(__name__)

DEFAULT_ERRORS: dict[str, str] = {"on_decode_failure": "silent_fail"}


class ParquetImageLoaderConfig(DataLoaderConfig):
    """The parquet image-rows loader config: generic keys plus decode specifics.

    ``type`` is pinned to ``parquet``; ``decode`` and ``image_shape`` are the
    image-specific keys the generic core config leaves to the plugin.

    Attributes:
        type: Pinned to ``parquet``.
        decode: Image payload form; only ``img_bytes`` is implemented
            (``rgba`` input is postponed).
        image_shape: Optional ``[height, width, channels]`` for image tables
            whose per-row shape is not carried in the data.
    """

    type: Literal["parquet"] = "parquet"
    decode: Literal["img_bytes", "rgba"] | None = Field(
        default=None,
        description="Image payload form; only 'img_bytes' is implemented.",
    )
    image_shape: list[int] | None = Field(
        default=None,
        description="Image shape [height, width, channels] when not carried in the data.",
    )


def _fnmatch_to_regex(pattern: str) -> str:
    """Convert an fnmatch pattern to a fully anchored, pyarrow-safe regex.

    ``fnmatch.translate`` emits ``(?s:...)\\Z``; pyarrow's RE2 rejects ``\\Z``
    (it wants ``\\z``) and search semantics would drop the start anchor, so the
    translated body is re-wrapped as ``^...$`` — full-string match, as fnmatch
    intends. Only ``*``, ``?`` and ``[...]`` patterns reach here
    (:func:`has_pattern`).

    Args:
        pattern: fnmatch pattern with ``*`` and ``?`` wildcards.

    Returns:
        A fully anchored regex string.
    """
    import fnmatch

    translated = fnmatch.translate(pattern)
    if translated.startswith("(?s:") and translated.endswith(")\\Z"):
        translated = translated[4:-3]
    return f"^({translated})$"


def _match_wildcard_arrow(col_expr: Any, patterns: list[str]) -> Any:
    """Return a pyarrow expression matching any of the fnmatch patterns.

    Args:
        col_expr: pyarrow field expression.
        patterns: List of fnmatch patterns.

    Returns:
        pyarrow compute expression for the OR of all patterns.
    """
    regex_patterns = [_fnmatch_to_regex(p) for p in patterns]
    combined_regex = "|".join(f"({p})" for p in regex_patterns)
    return pc.match_substring_regex(col_expr, combined_regex)


def _resolve_pyarrow_type(type_name: str) -> Any:
    """Map a ``TransformType`` string to a pyarrow data type.

    Args:
        type_name: One of ``int32``, ``int64``, ``float32``, ``float64``,
            ``bool``, ``str``, ``categorical``.

    Returns:
        The matching pyarrow data type.
    """
    mapping = {
        "int32": pa.int32(),
        "int64": pa.int64(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
        "str": pa.utf8(),
        "categorical": pa.dictionary(pa.int32(), pa.utf8()),
    }
    return mapping[type_name]


class ParquetDataSelection(DataSelection):
    """A selection of image rows from a Parquet dataset.

    One selection per loader, named ``<loader>``. Iteration yields one
    :class:`Batch` (batch = row) per ``load_batch_size`` chunk, with the image
    axes and value range fixed inline on the generic batch.

    Attributes:
        name: Selection name (``<loader>``).
        dataset: Name of the owning dataloader.
        image_column: The ``sample_path.column`` image column of each row.
        sample_axes: ``("height", "width", "channel")`` image layout.
        value_range: ``(0.0, 255.0)``.
    """

    sample_axes: tuple[str, ...] = ("height", "width", "channel")
    value_range: tuple[float, float] = (0.0, 255.0)

    def __init__(
        self,
        name: str,
        path: str,
        load_batch_size: int = 10000,
        threads: int = 4,
        filters_dict: dict[str, Any] | None = None,
        sample_path: list[dict[str, Any]] | None = None,
        transforms: list[dict[str, Any]] | None = None,
        id_column: str | None = None,
        image_shape: list[int] | None = None,
        errors: dict[str, str] | None = None,
        dataset: str | None = None,
    ):
        """Build a parquet image selection.

        Args:
            name: Selection name.
            path: Path to the parquet file or dataset directory.
            load_batch_size: Rows stepped from the table per batch.
            threads: Number of reader threads.
            filters_dict: Optional column → values filter mapping.
            sample_path: ``[{column, prefix}]``; the first entry names the
                image column and its ``prefix`` for relative-path rows.
            transforms: Column cast configurations.
            id_column: Column used as row identifier.
            image_shape: ``[height, width, channels]`` shared by all rows of a
                bytes column; nullable when ``height``/``width`` columns exist.
            errors: Row-decode error policy.
            dataset: Name of the owning dataloader; defaults to ``name``.
        """
        self.name = name
        self.dataset = dataset or name
        self.path = path
        self.load_batch_size = load_batch_size
        self.threads = threads
        self.filters_dict = filters_dict
        self.sample_path = sample_path or []
        self.transforms = transforms or []
        self.id_column = id_column
        self.image_shape = tuple(image_shape) if image_shape else None
        self.errors = {**DEFAULT_ERRORS, **(errors or {})}
        self.columns_list: list[str] | None = None
        self.filter_columns: list[str] = []
        self.filter_expr = None
        self._parquet: pq.ParquetDataset | None = None
        self.samples_count = 0
        self._failures = 0

        per_column_cfg = next((e for e in self.sample_path if e.get("column")), None)
        self.image_column = per_column_cfg["column"] if per_column_cfg else None
        self.image_prefix = per_column_cfg.get("prefix") if per_column_cfg else None

    @property
    def failures(self) -> int:
        """Number of rows skipped because an image row could not be decoded."""
        return self._failures

    @property
    def nb_samples(self) -> int:
        """Estimated number of rows in this selection."""
        return self.samples_count

    def _build_filter_expr(self) -> Any:
        """Build a pyarrow filter expression from the filters dictionary.

        Combines all filter conditions with AND logic.

        Returns:
            The pyarrow filter expression, or ``None`` when no filter is set.
        """
        if self.filters_dict is None:
            return None
        expr = None
        for col, val in self.filters_dict.items():
            if col not in self.filter_columns:
                self.filter_columns.append(col)
            from kcai_data_sampling_job.dataloaders.filters import build_filter_condition

            col_expr = build_filter_condition(
                col,
                val,
                wildcard_fn=lambda c, vals: _match_wildcard_arrow(pc.field(c), vals),
                isin_fn=lambda c, vals: pc.is_in(pc.field(c), pa.array(vals)),
                equal_fn=lambda c, v: pc.equal(pc.field(c), v),
            )
            expr = col_expr if expr is None else (expr & col_expr)
        return expr

    @override
    def bootstrap(self, columns_list: list[str] | None) -> None:
        """Initialize the parquet dataset and filter expression.

        Args:
            columns_list: Column names to load; ``None`` reads everything.
        """
        self.columns_list = columns_list or None
        self.filter_expr = self._build_filter_expr()
        self._parquet = pq.ParquetDataset(self.path, filters=self.filter_expr)
        if len(self._parquet.fragments) > 0:
            self.samples_count = sum(p.count_rows() for p in self._parquet.fragments)
        else:
            self.samples_count = 0

    def _read_columns(self) -> list[str] | None:
        """The columns to read: any explicit list, plus what decoding needs.

        Filter, image and id columns must always be in the read set; with no
        explicit ``columns_list`` everything is read.

        Returns:
            The read column list, or ``None`` for all columns.
        """
        if self.columns_list is None:
            return None
        required = [c for c in [*self.filter_columns, self.image_column, self.id_column] if c]
        merged = list(self.columns_list)
        for c in required:
            if c not in merged:
                merged.append(c)
        return merged

    def __len__(self) -> int:
        """Number of rows estimated for this selection."""
        return int(self.samples_count)

    @override
    def get_nb_batches(self) -> int:
        """Return the estimated number of batches.

        Returns:
            ``ceil(samples_count / load_batch_size)``.
        """
        return int(len(self) / self.load_batch_size) + (len(self) % self.load_batch_size > 0)

    def _apply_transforms(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Apply column casts to the batch.

        For each transform entry: ``in_place`` overwrites the column, other-
        wise a ``<column>_<to_type>`` column is appended.

        Args:
            batch: The pyarrow record batch.

        Returns:
            The cast record batch.
        """
        for t in self.transforms:
            col_idx = batch.schema.get_field_index(t["column"])
            if col_idx == -1:
                continue
            target_type = _resolve_pyarrow_type(t["to_type"])
            cast_col = batch.column(col_idx).cast(target_type)
            if t.get("in_place", False):
                batch = batch.set_column(col_idx, t["column"], cast_col)
            else:
                new_name = f"{t['column']}_{t['to_type']}"
                batch = batch.append_column(pa.field(new_name, target_type), cast_col)
        return batch

    def _fail_row(self) -> None:
        """Count a decode failure; raise when the policy is ``fail_fast``.

        Raises:
            RuntimeError: Re-raises the active exception under ``fail_fast``.
        """
        self._failures += 1
        if self.errors["on_decode_failure"] == "fail_fast":
            raise

    def _decode_rows(
        self,
        rows: list[Any],
        id_values: list[Any] | None,
        row_offset: int,
        resolve: Callable[[int, Any], tuple[int, int, bytes]],
    ) -> tuple[list[np.ndarray], list[str]]:
        """Decode raw payloads into arrays and row ids via ``resolve``.

        The shared per-row skeleton of the bytes and path paths: ``resolve``
        turns one row into ``(height, width, rgba payload)``; any exception in
        a row is counted as a decode failure (policy via ``_fail_row``) and
        drops the row, and ids fall back to the absolute row index when no id
        column is set.

        Args:
            rows: The per-row payloads to decode.
            id_values: Per-row identifiers, or ``None`` to fall back to the
                absolute row index.
            row_offset: Absolute index of the batch's first row (id fallback).
            resolve: Callable mapping ``(i, row)`` to the decoded height,
                width and raw rgba bytes.

        Returns:
            The decoded arrays and their row ids, in row order (failed rows
            dropped).
        """
        images: list[np.ndarray] = []
        ids: list[str] = []
        for i, row in enumerate(rows):
            try:
                height, width, payload = resolve(i, row)
                images.append(np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 4))
            except Exception:
                self._fail_row()
                continue
            ids.append(str(id_values[i]) if id_values is not None else str(row_offset + i))
        return images, ids

    def _decode_bytes(
        self,
        batch: pa.RecordBatch,
        img_values: list[Any],
        id_values: list[Any] | None,
        row_offset: int,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Decode a binary image column into arrays.

        Zero-copy ``np.frombuffer`` views; the per-row shape comes from
        ``image_shape`` or the ``height``/``width`` columns.

        Args:
            batch: The record batch (for ``height``/``width`` columns).
            img_values: Per-row image byte payloads.
            id_values: Per-row identifiers, or ``None`` to fall back to the
                absolute row index.
            row_offset: Absolute index of the batch's first row (id fallback).

        Returns:
            The decoded arrays and their row ids, in row order (failed rows
            dropped).
        """
        if self.image_shape is not None:
            shape = (int(self.image_shape[0]), int(self.image_shape[1]))
            h_col = w_col = None
        elif "height" in batch.column_names and "width" in batch.column_names:
            shape = None
            h_col = batch.column("height").to_numpy()
            w_col = batch.column("width").to_numpy()
        else:
            raise ValueError(
                "parquet bytes image column needs the row shape: set image_shape or provide 'height'/'width' columns."
            )

        def resolve(i: int, blob: Any) -> tuple[int, int, bytes]:
            if blob is None:
                raise ValueError("null image bytes row")
            if shape is not None:
                height, width = shape
            else:
                assert h_col is not None
                assert w_col is not None
                height, width = int(h_col[i]), int(w_col[i])
            return height, width, bytes(blob)

        return self._decode_rows(img_values, id_values, row_offset, resolve)

    def _decode_paths(
        self,
        img_values: list[Any],
        id_values: list[Any] | None,
        row_offset: int,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Decode a string image column (paths) into arrays.

        Absolute paths resolve directly; relative paths resolve against
        ``sample_path.prefix``. Every row decodes once through Pillow at load.

        Args:
            img_values: Per-row image path strings.
            id_values: Per-row identifiers, or ``None`` to fall back to the
                absolute row index.
            row_offset: Absolute index of the batch's first row (id fallback).

        Returns:
            The decoded arrays and their row ids, in row order (failed rows
            dropped).
        """

        def resolve(i: int, raw: Any) -> tuple[int, int, bytes]:
            if raw is None or not str(raw):
                raise ValueError("null image path row")
            value = str(raw)
            if Path(value).is_absolute():
                path = Path(value)
            elif self.image_prefix:
                path = Path(self.image_prefix) / value
            else:
                raise ValueError(f"relative image path {value!r} needs sample_path.prefix")
            height, width, rgba = decode_image(path)
            return height, width, rgba

        return self._decode_rows(img_values, id_values, row_offset, resolve)

    def _decode_batch(self, batch: pa.RecordBatch, row_offset: int) -> Batch:
        """Decode one chunk into a generic :class:`Batch` (batch = row).

        Decode contract: each image row decodes **once**, here, at load. The
        image column never reaches the batch — the remaining source columns
        stay arrow-native as the batch's ``columns``, and this loader fixes the
        image axes and value range on the generic batch's optional fields.

        Args:
            batch: The record batch to decode.
            row_offset: Absolute index of the batch's first row (id fallback).

        Returns:
            One ``Batch``: a row per decodable input row, with the image
            sample space declared.

        Raises:
            ValueError: If the image column is unset, missing, or neither
                binary nor string.
        """
        if self.image_column is None:
            raise ValueError(
                "parquet image mode needs an image column; set sample_path: [{column, prefix}] "
                "in the dataloader config."
            )
        if self.image_column not in batch.column_names:
            raise ValueError(f"parquet: image column {self.image_column!r} not found in the table.")

        img_values = batch.column(self.image_column).to_pylist()
        field_type = batch.schema.field(self.image_column).type
        is_bytes = pa.types.is_binary(field_type) or pa.types.is_large_binary(field_type)
        is_path = pa.types.is_string(field_type) or pa.types.is_large_string(field_type)
        if not (is_bytes or is_path):
            raise ValueError(
                f"parquet: image column {self.image_column!r} must be binary (raw bytes) or "
                f"string (path); got {field_type}."
            )

        id_values = (
            batch.column(self.id_column).to_pylist()
            if self.id_column and self.id_column in batch.column_names
            else None
        )

        images, ids = (
            self._decode_bytes(batch, img_values, id_values, row_offset)
            if is_bytes
            else self._decode_paths(img_values, id_values, row_offset)
        )
        return Batch(
            name=self.name,
            dataset=self.dataset,
            ids=ids,
            columns=pa.Table.from_batches([batch]).drop([self.image_column]),
            data=np.stack(images) if images else np.zeros((0, 0, 0, 0), dtype=np.uint8),
            sample_axes=self.sample_axes,
            value_range=self.value_range,
        )

    @override
    def __iter__(self) -> Iterator[Batch]:
        if self._parquet is None:
            return
        row_offset = 0
        for file in self._parquet.files:
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(
                batch_size=self.load_batch_size,
                columns=self._read_columns(),
                use_threads=self.threads,
            ):
                if self.filter_expr is not None:
                    batch = batch.filter(self.filter_expr)
                if len(batch) == 0:
                    continue
                batch = self._apply_transforms(batch)
                yield self._decode_batch(batch, row_offset)
                row_offset += len(batch)

    @override
    def __repr__(self) -> str:
        return f"ParquetSelection(name='{self.name}', path='{self.path}', filters={self.filters_dict})"


class ParquetDataLoader(DataLoader):
    """Data loader for parquet image tables (path or bytes columns).

    Generates the single selection named ``<loader>`` — one selection per
    loader, no split. Registered as the ``parquet`` type.

    Attributes:
        type: Loader type identifier.
        Config: The plugin's own config schema, resolved at config load.
    """

    type: str = "parquet"
    Config = ParquetImageLoaderConfig

    def __init__(
        self,
        name: str,
        config: ParquetImageLoaderConfig,
        errors: dict[str, str] | None = None,
        threads: int = 4,
    ):
        """Build the parquet loader.

        Args:
            name: Unique loader name.
            config: The plugin's resolved loader configuration; ``decode``
                must be ``"img_bytes"``.
            errors: Row-decode error policy override.
            threads: Number of reader threads.

        Raises:
            ValueError: If ``decode`` is not ``"img_bytes"``.
        """
        if config.decode and config.decode != "img_bytes":
            raise ValueError(f"parquet: decode mode {config.decode!r} is not implemented yet.")
        self.name = name
        self.config = config
        self.path = config.path
        self.load_batch_size = config.load_batch_size
        self.threads = threads
        self.filters_dict: dict[str, Any] = {}
        for item in config.filters or []:
            self.filters_dict[item.column] = item.values
        self.id_column = config.id_column
        self.sample_path = [sp.model_dump() for sp in config.sample_path or []]
        self.transforms = [t.model_dump() for t in config.transform or []]
        self.image_shape = config.image_shape
        self.errors = errors

    @override
    def get_selections(self) -> list[DataSelection]:
        """Create the loader's single selection.

        Returns:
            One ``ParquetDataSelection``, named ``<loader>``, covering the
            whole (filtered) dataset.
        """
        return [
            ParquetDataSelection(
                name=self.name,
                path=self.path,
                load_batch_size=self.load_batch_size,
                threads=self.threads,
                filters_dict=self.filters_dict,
                sample_path=self.sample_path,
                transforms=self.transforms,
                id_column=self.id_column,
                image_shape=self.image_shape,
                errors=self.errors,
                dataset=self.name,
            )
        ]
