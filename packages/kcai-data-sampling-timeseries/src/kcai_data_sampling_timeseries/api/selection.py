"""What the time-series packages feed the generic engine: one batch of windows.

:class:`TimeSeriesBatch` is the time-series specialization of the generic core
:class:`~kcai_data_sampling_core.api.selection.Batch`. The sample is a
**window** of ``W`` consecutive steps over ``C`` channels, so the stack is
``(B, W, C)`` and axis 1 is time. A signal has no fixed range of values:
``value_range`` stays ``None``, so the base neither clips nor refuses on range.
Row ``i`` on every axis is the same window: ``ids[i]``, ``columns`` slice
``i``, ``data[i]``.
"""

from dataclasses import dataclass

from kcai_data_sampling_core.api.selection import Batch
from typing_extensions import override


@dataclass
class TimeSeriesBatch(Batch):
    """One chunk of windows: the time-series specialization of :class:`Batch`.

    Attributes:
        name: Unique name of the selection this batch belongs to.
        dataset: Names the reader (loader) that assembled the selection.
        ids: One identifier per window, in row order.
        columns: The chunk's source columns as a pyarrow table — the values
            column is dropped — kept arrow-native for ledger pass-through.
        data: The window stack ``(B, W, C)``, floating point, one row per
            window, in row order.
        sample_axes: Labels of the window dimensions ``(W, C)``.
        value_range: ``None``: a signal is unbounded unless the data says so.
        sampling_rate: Steps per second of the source series, in Hz, or
            ``None`` when undeclared. Informative: algorithm parameters are in
            cycles per step, so no algorithm reads it.
    """

    sample_axes: tuple[str, ...] = ("time", "channel")
    value_range: tuple[float, float] | None = None
    sampling_rate: float | None = None

    @override
    def row(self, index: int) -> "TimeSeriesBatch":
        """Return a single-row batch: the window at ``index``.

        Args:
            index: Row index into every axis of the batch.

        Returns:
            The one-row batch.
        """
        return TimeSeriesBatch(
            name=self.name,
            dataset=self.dataset,
            ids=[self.ids[index]],
            columns=self.columns.slice(index, 1),
            data=self.data[index : index + 1],
            sample_axes=self.sample_axes,
            value_range=self.value_range,
            sampling_rate=self.sampling_rate,
        )
