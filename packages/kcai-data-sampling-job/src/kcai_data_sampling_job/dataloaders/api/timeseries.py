"""The ``timeseries_parquet`` dataloader: series rows in, window batches out.

Parquet is the only input. Each row is one **series**: an id, a values column
holding the whole recording (``list<float>`` for one channel, or
``list<list<float>>`` for ``T`` steps of ``C`` channels), and any other source
columns. The sample is not the row but a **window** of ``W`` consecutive
steps cut from it, and the selection decides how windows are cut:

- ``consecutive`` (:class:`ConsecutiveTimeSeriesSamples`): every window that
  fits, ``stride`` steps apart (``stride = 1`` slides, ``stride = W`` tiles);
- ``independent`` (:class:`IndependentTimeSeriesSamples`): ``n_windows``
  positions drawn per series, overlapping or not. The draw is seeded by
  ``seed`` and the series id, so the selection is the same whatever the read
  order or the batch size.

``channels`` (optional) keeps only the listed channels of each series, in
that order: the sample is ``(W, len(channels))``, and a transformation only
ever sees the selected channels. Each ``load_batch_size`` windows become one
generic core :class:`Batch` ``(B, W, C)`` float64, with the time-series space declared inline
(``("time", "channel")``, no value range) — ``-job`` depends on core only.
The source columns of the series are repeated on each of its windows, plus
``start_window`` and ``end_window`` (first and last step, inclusive). The
column named by ``t0_column``, when set, is carried as ``t0_series``: the
series' step 0, counted in steps from the step 0 of the earliest series (which
has ``t0_series = 0``). ``series_id``, ``t0_series``, ``start_window`` and
``end_window`` are the window's key: every relation between two windows (same
series, overlap, order and lag, simultaneity) is computed from it, in steps,
never stored. A window's id is opaque, a hash of ``(series, start, end)``:
nothing reads relations from it. Overlap is a
property of the selection: overlapping windows are correlated samples, and
the transformation never knows.
"""

from collections.abc import Iterator
import hashlib
import json
import logging
from typing import Annotated, Any, Literal, Self

from kcai_data_sampling_core.api.dataloaders import DataLoader, DataSelection
from kcai_data_sampling_core.api.selection import Batch
from kcai_data_sampling_core.models.dataloaders import DataLoaderConfig
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import override

from kcai_data_sampling_job.dataloaders.api.parquet import DEFAULT_ERRORS, _match_wildcard_arrow
from kcai_data_sampling_job.dataloaders.filters import build_filter_condition

logger = logging.getLogger(__name__)

#: The window space every time-series selection declares.
SAMPLE_AXES = ("time", "channel")

#: Rows per Arrow read batch; output batches are governed by ``load_batch_size``.
ARROW_ROW_BATCH_SIZE = 1024


class ConsecutiveWindowsConfig(BaseModel):
    """Every window that fits, ``stride`` steps apart.

    Attributes:
        mode: Pinned to ``consecutive``.
        window: Window length ``W`` in steps.
        stride: Steps between two window starts; ``None`` means ``W`` (windows
            tile the series without overlap), ``1`` slides.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["consecutive"] = "consecutive"
    window: int = Field(gt=0, description="Window length W, in steps.")
    stride: int | None = Field(default=None, gt=0, description="Steps between two starts; defaults to W.")


class IndependentWindowsConfig(BaseModel):
    """``n_windows`` window positions drawn per series.

    Attributes:
        mode: Pinned to ``independent``.
        window: Window length ``W`` in steps.
        n_windows: Windows drawn per series; never the same start twice.
        non_overlap: Forbid two windows of a series from sharing a step.
        seed: Seed of the draw, combined with each series id.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["independent"] = "independent"
    window: int = Field(gt=0, description="Window length W, in steps.")
    n_windows: int = Field(gt=0, description="Windows drawn per series.")
    non_overlap: bool = Field(default=False, description="Forbid windows of a series from sharing a step.")
    seed: int = Field(default=0, description="Seed of the draw, combined with each series id.")


WindowsConfig = Annotated[ConsecutiveWindowsConfig | IndependentWindowsConfig, Field(discriminator="mode")]


class TimeSeriesLoaderConfig(DataLoaderConfig):
    """The series-rows loader config: generic keys plus windowing.

    Attributes:
        type: Pinned to ``timeseries_parquet``.
        values_column: The column holding each series' values.
        t0_column: The integer column holding each series' step 0, in steps
            from the step 0 of the earliest series; carried as ``t0_series``.
            ``None`` when the series share no clock.
        channels: Indices of the channels each window keeps, in that order,
            distinct and ``>= 0``; ``None`` keeps them all.
        windows: How windows are cut from each series.
    """

    type: Literal["timeseries_parquet"] = "timeseries_parquet"
    values_column: str = Field(description="Column holding each series: list<float> or list<list<float>>.")
    t0_column: str | None = Field(default=None, description="Integer column holding step 0, in steps; optional.")
    channels: list[Annotated[int, Field(ge=0)]] | None = Field(
        default=None, min_length=1, description="Indices of the channels each window keeps; all of them when unset."
    )
    windows: WindowsConfig = Field(description="How windows are cut from each series.")

    @model_validator(mode="after")
    def _no_image_keys(self) -> Self:
        """Refuse the generic keys this loader does not honor.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``sample_path`` or ``transform`` is set, or if
                ``channels`` lists a channel twice.
        """
        for key in ("sample_path", "transform"):
            if getattr(self, key) is not None:
                raise ValueError(f"timeseries_parquet: {key!r} is not supported")
        if self.channels is not None and len(set(self.channels)) != len(self.channels):
            raise ValueError(f"timeseries_parquet: channels must be distinct, got {self.channels}")
        return self


def _series_seed(seed: int, series_id: str) -> list[int]:
    """The draw seed of one series: the selection seed and a hash of its id.

    Args:
        seed: The selection's ``seed``.
        series_id: The series identifier.

    Returns:
        The seed sequence for ``np.random.default_rng``.
    """
    return [seed, int.from_bytes(hashlib.sha1(series_id.encode()).digest()[:8], "big")]


def window_id(series_id: str, start: int, end: int) -> str:
    """The opaque id of one window: a 12-hex sha1 of its key.

    Args:
        series_id: The series identifier.
        start: The window's first step.
        end: The window's last step (inclusive).

    Returns:
        The id, the same for the same window in every run and mode.
    """
    key = json.dumps([series_id, start, end])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def consecutive_starts(length: int, window: int, stride: int) -> list[int]:
    """Every start whose window fits in a series of ``length`` steps.

    Args:
        length: Steps in the series.
        window: Window length ``W``.
        stride: Steps between two starts.

    Returns:
        The starts ``0, stride, 2 * stride, ...``; empty when the series is
        shorter than ``W``.
    """
    return list(range(0, length - window + 1, stride))


def independent_starts(length: int, config: IndependentWindowsConfig, series_id: str) -> list[int]:
    """Draw ``n_windows`` distinct starts for one series, sorted.

    Without overlap, the draw is uniform over the placements of ``N``
    disjoint windows: ``N`` distinct values ``p`` among ``length - N * (W - 1)``
    positions, sorted, become starts ``p_i + i * (W - 1)``, so consecutive
    windows are at least ``W`` steps apart.

    Args:
        length: Steps in the series.
        config: The independent-windows settings.
        series_id: The series identifier, combined with the seed.

    Returns:
        The sorted starts.

    Raises:
        ValueError: If ``n_windows`` windows of length ``W`` cannot be placed
            in the series under the overlap setting.
    """
    n, window = config.n_windows, config.window
    rng = np.random.default_rng(_series_seed(config.seed, series_id))
    positions = length - n * (window - 1) if config.non_overlap else length - window + 1
    if positions < n:
        overlap = "non-overlapping" if config.non_overlap else "distinct"
        raise ValueError(f"series {series_id!r}: {n} {overlap} windows of {window} steps do not fit in {length}")
    picks = np.sort(rng.choice(positions, size=n, replace=False))
    if config.non_overlap:
        picks = picks + np.arange(n) * (window - 1)
    return [int(p) for p in picks]


def as_series(values: Any) -> np.ndarray:
    """Turn one row's values into a ``(T, C)`` float64 array.

    Args:
        values: A list of floats, or a list of per-step channel lists.

    Returns:
        The series, one row per step.

    Raises:
        ValueError: If the values are missing, not numeric, or not 1-D/2-D.
    """
    if values is None:
        raise ValueError("null series row")
    series = np.asarray(values, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, None]
    if series.ndim != 2:
        raise ValueError(f"a series must be list<float> or list<list<float>>, got {series.ndim} dimensions")
    return series


def as_series_arrow(column: pa.Array | pa.ChunkedArray, index: int) -> np.ndarray:
    """Read one row of an Arrow values column as a ``(T, C)`` float64 array.

    Supported schemas (``list<double>``, ``list<list<double>>`` and the fixed
    inner lists written by ``fetch_ett_sample.py``) are flattened from Arrow
    offsets and buffers without going through Python lists. Anything else falls
    back to :func:`as_series`.

    Args:
        column: The values column of one Arrow record batch.
        index: Row index within that column.

    Returns:
        The series, one row per step.

    Raises:
        ValueError: If the row is null, ragged, or not a 1-D/2-D series.
    """
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if array.is_null()[index].as_py():
        raise ValueError("null series row")
    if not pa.types.is_list(array.type):
        return as_series(array[index].as_py())
    item_type = array.values.type
    if not (pa.types.is_floating(item_type) or pa.types.is_list(item_type) or pa.types.is_fixed_size_list(item_type)):
        return as_series(array[index].as_py())

    offsets = array.offsets.to_numpy(zero_copy_only=False)
    first, last = int(offsets[index]), int(offsets[index + 1])
    item = array.values
    if pa.types.is_floating(item.type):
        values = np.asarray(item.to_numpy(zero_copy_only=False)[first:last], dtype=np.float64)
        return values[:, None]
    if pa.types.is_fixed_size_list(item.type):
        width = item.type.list_size
        values = np.asarray(item.values.to_numpy(zero_copy_only=False)[first * width : last * width], dtype=np.float64)
        return values.reshape(-1, width)

    inner_offsets = item.offsets.to_numpy(zero_copy_only=False)
    widths = np.diff(inner_offsets[first : last + 1])
    if widths.size == 0 or not np.all(widths == widths[0]):
        raise ValueError("a series must have a constant channel count")
    width = int(widths[0])
    starts = inner_offsets[first:last]
    indices = (starts[:, None] + np.arange(width)).ravel()
    values = np.asarray(item.values.to_numpy(zero_copy_only=False)[indices], dtype=np.float64)
    return values.reshape(-1, width)


class TimeSeriesDataSelection(DataSelection):
    """A selection of windows cut from the series of a Parquet dataset.

    One selection per loader, named ``<loader>``. Iteration yields one
    :class:`Batch` per ``load_batch_size`` windows; the last one may be
    shorter.

    Attributes:
        name: Selection name (``<loader>``).
        dataset: Name of the owning dataloader.
        sample_axes: ``("time", "channel")``.
        value_range: ``None``: a signal is unbounded.
    """

    sample_axes: tuple[str, ...] = SAMPLE_AXES
    value_range: tuple[float, float] | None = None

    def __init__(
        self,
        name: str,
        config: TimeSeriesLoaderConfig,
        threads: int = 4,
        errors: dict[str, str] | None = None,
    ):
        """Build a time-series selection.

        Args:
            name: Selection name.
            config: The loader's resolved configuration.
            threads: Number of reader threads.
            errors: Row error policy (``on_decode_failure``).
        """
        self.name = name
        self.dataset = name
        self.config = config
        self.windows = config.windows
        self.load_batch_size = config.load_batch_size
        self.threads = threads
        self.errors = {**DEFAULT_ERRORS, **(errors or {})}
        self.filter_expr: Any = None
        self._parquet: pq.ParquetDataset | None = None
        self._lengths: list[int] = []
        self._channels: int | None = None
        self._failures = 0

    @property
    def failures(self) -> int:
        """Number of series skipped because they could not be windowed."""
        return self._failures

    @property
    def nb_samples(self) -> int:
        """Number of series in this selection (the failure-rate denominator)."""
        return len(self._lengths)

    def _build_filter_expr(self) -> Any:
        """Build the pyarrow filter expression from ``filters`` (AND of all).

        Returns:
            The expression, or ``None`` when no filter is set.
        """
        expr = None
        for item in self.config.filters or []:
            condition = build_filter_condition(
                item.column,
                item.values,
                wildcard_fn=lambda c, vals: _match_wildcard_arrow(pc.field(c), vals),
                isin_fn=lambda c, vals: pc.is_in(pc.field(c), pa.array(vals)),
                equal_fn=lambda c, v: pc.equal(pc.field(c), v),
            )
            expr = condition if expr is None else (expr & condition)
        return expr

    @override
    def bootstrap(self, columns_list: list[str] | None) -> None:
        """Open the dataset and read each series' length, for the batch count.

        Args:
            columns_list: Ignored: every source column is carried to the
                windows of its series.
        """
        del columns_list
        self.filter_expr = self._build_filter_expr()
        self._parquet = pq.ParquetDataset(self.config.path, filters=self.filter_expr)
        self._check_t0_column(self._parquet.schema)
        if "n_steps" in self._parquet.schema.names:
            lengths = self._parquet.read(["n_steps"]).column("n_steps").to_pylist()
            self._lengths = [int(n or 0) for n in lengths]
        else:
            self._lengths = []
            for file in self._parquet.files:
                for batch in pq.ParquetFile(file).iter_batches(
                    batch_size=ARROW_ROW_BATCH_SIZE,
                    columns=[self.config.values_column],
                    use_threads=self.threads,
                ):
                    counts = pc.list_value_length(batch.column(0))
                    self._lengths.extend(int(n or 0) for n in counts.to_pylist())

    def _check_t0_column(self, schema: pa.Schema) -> None:
        """Check that ``t0_column``, when set, is an integer column.

        Args:
            schema: The dataset schema.

        Raises:
            ValueError: If the column is missing or of another type.
        """
        name = self.config.t0_column
        if name is None:
            return
        if name not in schema.names:
            raise ValueError(f"timeseries_parquet: t0_column {name!r} is not in the dataset")
        kind = schema.field(name).type
        if not pa.types.is_integer(kind):
            raise ValueError(f"timeseries_parquet: t0_column {name!r} must be an integer step count, got {kind}")

    def _starts(self, length: int, series_id: str) -> list[int]:
        """The window starts of one series, per the selection's mode.

        Args:
            length: Steps in the series.
            series_id: The series identifier.

        Returns:
            The sorted starts.
        """
        if isinstance(self.windows, IndependentWindowsConfig):
            return independent_starts(length, self.windows, series_id)
        return consecutive_starts(length, self.windows.window, self.windows.stride or self.windows.window)

    @override
    def get_nb_batches(self) -> int:
        """Return the estimated number of batches, from the series lengths.

        Returns:
            ``ceil(windows / load_batch_size)``; an independent selection is
            counted at ``n_windows`` per series.
        """
        window = self.windows.window
        if isinstance(self.windows, IndependentWindowsConfig):
            total = self.windows.n_windows * len(self._lengths)
        else:
            stride = self.windows.stride or window
            total = sum(len(range(0, n - window + 1, stride)) for n in self._lengths)
        return -(-total // self.load_batch_size)

    def _fail_row(self) -> None:
        """Count a series that could not be windowed; raise under ``fail_fast``.

        Raises:
            Exception: Re-raises the active exception under ``fail_fast``.
        """
        self._failures += 1
        if self.errors["on_decode_failure"] == "fail_fast":
            raise

    def _cut(
        self, series_id: str, values: pa.Array | pa.ChunkedArray, index: int
    ) -> tuple[np.ndarray, list[int]] | None:
        """Cut one series into its windows, or count it as a failure.

        Args:
            series_id: The series identifier.
            values: The record batch's Arrow values column.
            index: The row's index in that column.

        A series shorter than one window yields no window and is not a
        failure; a series without one of the selected ``channels``, or with
        another channel count than the first one, is (the batch stacks
        windows of one shape).

        Returns:
            The windows ``(n, W, C)`` and their starts, or ``None`` when the
            series yields no window.
        """
        try:
            series = self._select_channels(as_series_arrow(values, index), series_id)
            if self._channels is not None and series.shape[1] != self._channels:
                raise ValueError(f"series {series_id!r} has {series.shape[1]} channels, expected {self._channels}")
            starts = self._starts(len(series), series_id)
        except ValueError:
            self._fail_row()
            return None
        self._channels = series.shape[1]
        if not starts:
            return None
        window = self.windows.window
        return np.stack([series[s : s + window] for s in starts]), starts

    def _select_channels(self, series: np.ndarray, series_id: str) -> np.ndarray:
        """Keep the selected channels of one series, in the listed order.

        Args:
            series: The series, ``(T, C)``.
            series_id: The series identifier, for the error message.

        Returns:
            ``(T, len(channels))``, or the series itself when no channel is
            selected.

        Raises:
            ValueError: If a selected channel does not exist in the series.
        """
        channels = self.config.channels
        if channels is None:
            return series
        missing = [c for c in channels if c >= series.shape[1]]
        if missing:
            raise ValueError(f"series {series_id!r} has {series.shape[1]} channels, no channel {missing}")
        return series[:, channels]

    def _windows_of(
        self, record: pa.RecordBatch, row_offset: int, row_indices: list[int] | None = None
    ) -> Iterator[tuple[np.ndarray, pa.Table]]:
        """Window every series of one record batch.

        Args:
            record: The chunk of series rows (already batch-filtered).
            row_offset: Absolute index of the chunk's first row in the row order
                of the retained files, before batch-level predicate filtering.
            row_indices: Original row indices of the surviving rows, used for
                fallback ids. Defaults to the unfiltered chunk's positions.

        Yields:
            Per series: its windows ``(n, W, C)`` and their source columns.
        """
        id_column = self.config.id_column
        if id_column:
            row_ids = record.column(id_column).to_pylist()
        else:
            row_ids = row_indices if row_indices is not None else range(row_offset, row_offset + len(record))
        values = record.column(self.config.values_column)
        sources = pa.Table.from_batches([record]).drop([self.config.values_column])
        if self.config.t0_column is not None:
            t0_series = sources.column(self.config.t0_column).cast(pa.int64())
            sources = sources.drop([self.config.t0_column]).append_column("t0_series", t0_series)
        for i, series_id in enumerate(row_ids):
            cut = self._cut(str(series_id), values, i)
            if cut is None:
                continue
            windows, starts = cut
            ends = [s + self.windows.window - 1 for s in starts]
            columns = sources.slice(i, 1).take([0] * len(starts))
            columns = columns.append_column("start_window", pa.array(starts, pa.int64()))
            columns = columns.append_column("end_window", pa.array(ends, pa.int64()))
            ids = [window_id(str(series_id), s, e) for s, e in zip(starts, ends, strict=True)]
            columns = columns.append_column("window_id", pa.array(ids))
            yield windows, columns

    def _batch(self, data: np.ndarray, columns: pa.Table) -> Batch:
        """Wrap windows and their columns into one generic batch.

        Args:
            data: The window stack ``(B, W, C)``.
            columns: The windows' source columns, ``window_id`` included.

        Returns:
            The batch, ids taken from ``window_id`` (dropped from the columns).
        """
        return Batch(
            name=self.name,
            dataset=self.dataset,
            ids=columns.column("window_id").to_pylist(),
            columns=columns.drop(["window_id"]),
            data=data,
            sample_axes=self.sample_axes,
            value_range=self.value_range,
        )

    def _filter_record(self, record: pa.RecordBatch, row_offset: int) -> tuple[pa.RecordBatch, list[int]]:
        """Filter one Arrow record, returning survivors and their global row indices."""
        assert self.filter_expr is not None
        index_column = "_timeseries_row_index"
        while index_column in record.schema.names:
            index_column += "_"
        indexed = pa.Table.from_batches([record]).append_column(
            index_column, pa.array(range(record.num_rows), pa.int64())
        )
        filtered = indexed.filter(self.filter_expr)
        row_indices = [row_offset + i for i in filtered.column(index_column).to_pylist()]
        survivors = pa.RecordBatch.from_arrays(
            [filtered.column(name).combine_chunks() for name in record.schema.names],
            schema=record.schema,
        )
        return survivors, row_indices

    @override
    def __iter__(self) -> Iterator[Batch]:
        if self._parquet is None:
            return
        pending_data: list[np.ndarray] = []
        pending_columns: list[pa.Table] = []
        row_offset = 0
        for file in self._parquet.files:
            for record in pq.ParquetFile(file).iter_batches(batch_size=ARROW_ROW_BATCH_SIZE, use_threads=self.threads):
                raw_rows = record.num_rows
                if self.filter_expr is None:
                    row_indices = list(range(row_offset, row_offset + raw_rows))
                else:
                    record, row_indices = self._filter_record(record, row_offset)
                for windows, columns in self._windows_of(record, row_offset, row_indices):
                    pending_data.append(windows)
                    pending_columns.append(columns)
                row_offset += raw_rows
                yield from self._drain(pending_data, pending_columns, final=False)
        yield from self._drain(pending_data, pending_columns, final=True)

    def _drain(self, data: list[np.ndarray], columns: list[pa.Table], final: bool) -> Iterator[Batch]:
        """Emit full batches from the pending windows; the remainder at the end.

        Pending windows stay as one chunk per source record. Only the chunks
        consumed by a batch are concatenated, so emitted buffers are released
        and a tail is a slice of its original chunk, not of an accumulation.

        Args:
            data: Pending window stacks, consumed in place.
            columns: Pending window columns, consumed in place.
            final: Emit the last, possibly shorter, batch too.

        Yields:
            Batches of exactly ``load_batch_size`` windows, then the remainder
            when ``final``.
        """
        size = self.load_batch_size
        while data:
            available = sum(len(chunk) for chunk in data)
            if available < size and not final:
                break
            stack, batch_columns = self._take_pending(data, columns, min(size, available))
            yield self._batch(stack, batch_columns)

    def _take_pending(self, data: list[np.ndarray], columns: list[pa.Table], take: int) -> tuple[np.ndarray, pa.Table]:
        """Consume ``take`` windows from the pending chunks and concatenate them."""
        arrays: list[np.ndarray] = []
        tables: list[pa.Table] = []
        remaining = take
        while remaining:
            chunk, table = data[0], columns[0]
            n = len(chunk)
            if n <= remaining:
                arrays.append(chunk)
                tables.append(table)
                data.pop(0)
                columns.pop(0)
                remaining -= n
            else:
                arrays.append(chunk[:remaining])
                tables.append(table.slice(0, remaining))
                data[0] = chunk[remaining:]
                columns[0] = table.slice(remaining)
                remaining = 0
        stack = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
        batch_columns = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        return stack, batch_columns

    @override
    def __repr__(self) -> str:
        return f"TimeSeriesSelection(name='{self.name}', path='{self.config.path}', windows={self.windows.mode})"


class ConsecutiveTimeSeriesSamples(TimeSeriesDataSelection):
    """Windows ``stride`` steps apart, covering each series (``consecutive``)."""


class IndependentTimeSeriesSamples(TimeSeriesDataSelection):
    """``n_windows`` seeded window positions per series (``independent``)."""


class TimeSeriesParquetDataLoader(DataLoader):
    """Data loader for parquet tables of series, one row per series.

    Generates the single selection named ``<loader>``, of the class matching
    ``windows.mode``. Registered as the ``timeseries_parquet`` type.

    Attributes:
        type: Loader type identifier.
        Config: The plugin's own config schema, resolved at config load.
    """

    type: str = "timeseries_parquet"
    Config = TimeSeriesLoaderConfig

    def __init__(
        self,
        name: str,
        config: TimeSeriesLoaderConfig,
        errors: dict[str, str] | None = None,
        threads: int = 4,
    ):
        """Build the loader.

        Args:
            name: Unique loader name.
            config: The plugin's resolved loader configuration.
            errors: Row error policy override.
            threads: Number of reader threads.
        """
        self.name = name
        self.config = config
        self.errors = errors
        self.threads = threads

    @override
    def get_selections(self) -> list[DataSelection]:
        """Create the loader's single selection.

        Returns:
            One selection, named ``<loader>``: consecutive or independent
            windows per ``windows.mode``.
        """
        selection_cls = (
            IndependentTimeSeriesSamples
            if isinstance(self.config.windows, IndependentWindowsConfig)
            else ConsecutiveTimeSeriesSamples
        )
        return [selection_cls(name=self.name, config=self.config, threads=self.threads, errors=self.errors)]
