"""The ``timeseries_parquet`` dataloader: series rows in, window batches out.

Parquet is the only input. Each row is one **series**: an id, a values column
holding the whole recording (``list<float>`` for one channel, or
``list<list<float>>`` for ``T`` steps of ``C`` channels), and any other source
columns. The sample is not the row but a **window** of ``W`` consecutive
steps cut from it, and the selection decides how windows are cut:

- ``consecutive`` (:class:`ConsecutiveTimeSeriesSamples`): every window that
  fits, ``stride`` steps apart (``stride = 1`` slides, ``stride = W`` tiles);
  a window's id is ``<series>@<t>``, ``t`` its last step (the lookback ending
  at ``t``);
- ``independent`` (:class:`IndependentTimeSeriesSamples`): ``n_windows``
  positions drawn per series, overlapping or not; a window's id is
  ``<series>@<start>``. The draw is seeded by ``seed`` and the series id, so
  the selection is the same whatever the read order or the batch size.

Each ``load_batch_size`` windows become one generic core :class:`Batch`
``(B, W, C)`` float64, with the time-series space declared inline
(``("time", "channel")``, no value range) — ``-job`` depends on core only.
The source columns of the series are repeated on each of its windows, plus
``window_start`` and ``window_end`` (first and last step, inclusive). Overlap
is a property of the selection: overlapping windows are correlated samples,
and the transformation never knows.
"""

from collections.abc import Iterator
import hashlib
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
        sampling_rate: Steps per second of the series, in Hz; informative.
        windows: How windows are cut from each series.
    """

    type: Literal["timeseries_parquet"] = "timeseries_parquet"
    values_column: str = Field(description="Column holding each series: list<float> or list<list<float>>.")
    sampling_rate: float | None = Field(default=None, gt=0, description="Sampling rate in Hz; informative.")
    windows: WindowsConfig = Field(description="How windows are cut from each series.")

    @model_validator(mode="after")
    def _no_image_keys(self) -> Self:
        """Refuse the generic keys this loader does not honor.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``sample_path`` or ``transform`` is set.
        """
        for key in ("sample_path", "transform"):
            if getattr(self, key) is not None:
                raise ValueError(f"timeseries_parquet: {key!r} is not supported")
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
        lengths = pc.list_value_length(self._parquet.read([self.config.values_column]).column(0))
        self._lengths = [int(n or 0) for n in lengths.to_pylist()]

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

    def _window_id(self, series_id: str, start: int) -> str:
        """The id of one window: ``<series>@<last step>`` or ``<series>@<start>``.

        Args:
            series_id: The series identifier.
            start: The window's first step.

        Returns:
            The window id.
        """
        if isinstance(self.windows, IndependentWindowsConfig):
            return f"{series_id}@{start}"
        return f"{series_id}@{start + self.windows.window - 1}"

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

    def _cut(self, series_id: str, values: Any) -> tuple[np.ndarray, list[int]] | None:
        """Cut one series into its windows, or count it as a failure.

        Args:
            series_id: The series identifier.
            values: The row's raw values.

        A series shorter than one window yields no window and is not a
        failure; a series with another channel count than the first one is
        (the batch stacks windows of one shape).

        Returns:
            The windows ``(n, W, C)`` and their starts, or ``None`` when the
            series yields no window.
        """
        try:
            series = as_series(values)
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

    def _windows_of(self, record: pa.RecordBatch, row_offset: int) -> Iterator[tuple[np.ndarray, pa.Table]]:
        """Window every series of one record batch.

        Args:
            record: The chunk of series rows.
            row_offset: Absolute index of the chunk's first row (id fallback).

        Yields:
            Per series: its windows ``(n, W, C)`` and their source columns.
        """
        id_column = self.config.id_column
        ids = record.column(id_column).to_pylist() if id_column else range(row_offset, row_offset + len(record))
        values = record.column(self.config.values_column).to_pylist()
        sources = pa.Table.from_batches([record]).drop([self.config.values_column])
        for i, (series_id, row_values) in enumerate(zip(ids, values, strict=True)):
            cut = self._cut(str(series_id), row_values)
            if cut is None:
                continue
            windows, starts = cut
            columns = sources.slice(i, 1).take([0] * len(starts))
            columns = columns.append_column("window_start", pa.array(starts, pa.int64()))
            columns = columns.append_column(
                "window_end", pa.array([s + self.windows.window - 1 for s in starts], pa.int64())
            )
            columns = columns.append_column("window_id", pa.array([self._window_id(str(series_id), s) for s in starts]))
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

    @override
    def __iter__(self) -> Iterator[Batch]:
        if self._parquet is None:
            return
        pending_data: list[np.ndarray] = []
        pending_columns: list[pa.Table] = []
        row_offset = 0
        for file in self._parquet.files:
            for record in pq.ParquetFile(file).iter_batches(batch_size=self.load_batch_size, use_threads=self.threads):
                if self.filter_expr is not None:
                    record = record.filter(self.filter_expr)
                for windows, columns in self._windows_of(record, row_offset):
                    pending_data.append(windows)
                    pending_columns.append(columns)
                row_offset += len(record)
                yield from self._drain(pending_data, pending_columns, final=False)
        yield from self._drain(pending_data, pending_columns, final=True)

    def _drain(self, data: list[np.ndarray], columns: list[pa.Table], final: bool) -> Iterator[Batch]:
        """Emit full batches from the pending windows; the remainder at the end.

        Args:
            data: Pending window stacks, consumed in place.
            columns: Pending window columns, consumed in place.
            final: Emit the last, possibly shorter, batch too.

        Yields:
            Batches of exactly ``load_batch_size`` windows, then the remainder
            when ``final``.
        """
        if not data:
            return
        stack = np.concatenate(data)
        table = pa.concat_tables(columns)
        size = self.load_batch_size
        cut = len(stack) if final else len(stack) - len(stack) % size
        for start in range(0, cut, size):
            stop = min(start + size, cut)
            yield self._batch(stack[start:stop], table.slice(start, stop - start))
        data[:] = [stack[cut:]] if cut < len(stack) else []
        columns[:] = [table.slice(cut)] if cut < len(stack) else []

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
