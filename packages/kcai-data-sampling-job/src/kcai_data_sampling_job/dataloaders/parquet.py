"""The ``parquet`` dataloader: a table of ``img_bytes`` rows becomes samples.

Image mode (phase 1): each row's image column holds raw RGBA bytes; the
selection decodes them on the fly into ``(H, W, 4)`` uint8 views. The row
shape comes from the ``image_shape`` config or ``height``/``width`` columns;
no pillow here (job stays codec-free). Filters, splits, casts and
path-prefix handling port from dqm-ml, with the batch-size rename carried over.

Tabular (non-decode) rows are refused this phase: there is no tabular
transformation yet.
"""

import logging
from pathlib import Path
from typing import Any, Iterator

from kcai_data_sampling_core.api.selection import Sample
from kcai_data_sampling_core.models.dataloaders import DataLoaderConfig, SplitConfig
from kcai_data_sampling_core.utils.matching import has_pattern, resolve_include_exclude, resolve_patterns
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from typing_extensions import override

from kcai_data_sampling_job.dataloaders.proto import DataLoader, DataSelection

logger = logging.getLogger(__name__)

DEFAULT_ERRORS: dict[str, str] = {"on_decode_failure": "silent_fail"}


def _fnmatch_to_regex(pattern: str) -> str:
    """Convert an fnmatch pattern to a regex pattern.

    Args:
        pattern: fnmatch pattern with ``*`` and ``?`` wildcards.

    Returns:
        Regex pattern string.
    """
    import fnmatch

    return fnmatch.translate(pattern)


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

    Attributes:
        name: Selection name (``{loader}`` or ``{loader}_{split_value}``).
        dataset: Name of the owning dataloader.
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
                image column for img_bytes decode.
            transforms: Column cast configurations.
            id_column: Column used as row identifier.
            image_shape: ``[height, width, channels]`` shared by all rows;
                nullable when ``height``/``width`` columns exist.
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
        self.filter_expr = None
        self.dataset: pq.ParquetDataset | None = None
        self.samples_count = 0
        self._failures = 0

        per_column_cfg = next((e for e in self.sample_path if e.get("column")), None)
        self.image_column = per_column_cfg["column"] if per_column_cfg else None

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
            if self.columns_list is None:
                self.columns_list = [col]
            elif col not in self.columns_list:
                self.columns_list.append(col)
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
        self.dataset = pq.ParquetDataset(self.path, filters=self.filter_expr)
        if len(self.dataset.fragments) > 0:
            self.samples_count = sum(p.count_rows() for p in self.dataset.fragments)
        else:
            self.samples_count = 0

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

    def _decode_batch(self, batch: pa.RecordBatch, row_offset: int) -> list[Sample]:
        """Decode one batch of ``img_bytes`` rows into sample objects.

        Global decode contract: each row's bytes are turned into numpy
        once, here, as a zero-copy ``np.frombuffer`` view.

        Args:
            batch: The record batch to decode.
            row_offset: Absolute index of the batch's first row (id fallback).

        Returns:
            One ``Sample`` per decodable row.

        Raises:
            RuntimeError: If the error policy is ``fail_fast`` and a row
                cannot be decoded.
        """
        if self.image_column is None:
            raise ValueError(
                "parquet image mode needs an image column; set sample_path: [{column: <img_bytes>}] "
                "in the dataloader config."
            )
        if self.image_column not in batch.column_names:
            raise ValueError(f"parquet: image column {self.image_column!r} not found in the table.")

        if self.image_shape is not None:
            shape_h, shape_w = int(self.image_shape[0]), int(self.image_shape[1])
            per_row_shape = True
        elif "height" in batch.column_names and "width" in batch.column_names:
            per_row_shape = False
            h_col = batch.column("height").to_numpy()
            w_col = batch.column("width").to_numpy()
        else:
            raise ValueError(
                "parquet image mode needs the row shape: set image_shape or provide 'height'/'width' "
                "columns."
            )

        img_bytes = batch.column(self.image_column).to_numpy(zero_copy_only=False)
        if self.id_column and self.id_column in batch.column_names:
            id_values = batch.column(self.id_column).to_pylist()
        else:
            id_values = None

        rows: list[Sample] = []
        for i, blob in enumerate(img_bytes):
            try:
                if blob is None:
                    raise ValueError("null img_bytes row")
                if per_row_shape:
                    height, width = shape_h, shape_w
                else:
                    height, width = int(h_col[i]), int(w_col[i])
                x = np.frombuffer(bytes(blob), dtype=np.uint8).reshape(height, width, 4)
            except Exception:
                self._failures += 1
                if self.errors["on_decode_failure"] == "fail_fast":
                    raise
                continue
            row_id = str(id_values[i]) if id_values is not None else str(row_offset + i)
            rows.append(
                Sample(
                    id=row_id,
                    x=x,
                    y={},
                    source={"column": self.image_column, "row": row_id},
                )
            )
        return rows

    @override
    def __iter__(self) -> Iterator[list[Sample]]:
        if self.dataset is None:
            return
        row_offset = 0
        for file in self.dataset.files:
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(
                batch_size=self.load_batch_size,
                columns=self.columns_list,
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
    """Data loader for parquet ``img_bytes`` tables.

    Generates one selection per split value (or a single one when no split is
    configured). Registered as the ``parquet`` type.

    Attributes:
        type: Loader type identifier.
    """

    type: str = "parquet"

    def __init__(
        self,
        name: str,
        config: DataLoaderConfig,
        errors: dict[str, str] | None = None,
        threads: int = 4,
    ):
        """Build the parquet loader.

        Args:
            name: Unique loader name.
            config: Dataloader configuration; ``decode`` must be
                ``"img_bytes"`` this phase.
            errors: Row-decode error policy override.
            threads: Number of reader threads.

        Raises:
            ValueError: If ``decode`` is not ``"img_bytes"``.
        """
        if config.decode and config.decode != "img_bytes":
            raise ValueError(
                f"parquet: decode mode {config.decode!r} is not implemented this phase."
            )
        self.name = name
        self.config = config
        self.path = config.path
        self.load_batch_size = config.load_batch_size
        self.threads = threads
        self.split = SplitConfig.model_validate(config.split) if config.split else None
        self.split_by = self.split.by if self.split else None
        self.split_values = self.split.values if self.split else None
        self.filters_dict: dict[str, Any] = {}
        for item in config.filters or []:
            self.filters_dict[item.column] = item.values
        self.id_column = config.id_column
        self.sample_path = [sp.model_dump() for sp in config.sample_path or []]
        self.transforms = [t.model_dump() for t in config.transform or []]
        self.image_shape = config.image_shape
        self.errors = errors

    def get_selections(self) -> list[DataSelection]:
        """Create the selections from the split configuration.

        Returns:
            One ``ParquetDataSelection`` per split value, or a single one
            covering the whole dataset.
        """
        path = self.path

        if not self.split_by:
            return [
                ParquetDataSelection(
                    name=self.name,
                    path=path,
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

        values = self.split_values
        if values is None:
            logger.info("Discovering unique values for split_by='%s' in %s", self.split_by, path)
            table = pq.read_table(path, columns=[self.split_by])
            values = [str(v) for v in pc.unique(table.column(0)).to_pylist() if v is not None]
        else:
            if any(has_pattern(v) for v in values):
                logger.info("Expanding wildcard values for split_by='%s' in %s", self.split_by, path)
                table = pq.read_table(path, columns=[self.split_by])
                available = [str(v) for v in pc.unique(table.column(0)).to_pylist() if v is not None]
                values = resolve_patterns(values, available)

        if self.split and self.split.exclude:
            values = resolve_include_exclude(None, self.split.exclude, values)

        selections: list[DataSelection] = []
        for val in values:
            selection_name = f"{self.name}_{val}"
            merged_filters = (self.filters_dict or {}).copy()
            merged_filters[self.split_by] = val
            selections.append(
                ParquetDataSelection(
                    name=selection_name,
                    path=path,
                    load_batch_size=self.load_batch_size,
                    threads=self.threads,
                    filters_dict=merged_filters,
                    sample_path=self.sample_path,
                    transforms=self.transforms,
                    id_column=self.id_column,
                    image_shape=self.image_shape,
                    errors=self.errors,
                    dataset=self.name,
                )
            )
        return selections