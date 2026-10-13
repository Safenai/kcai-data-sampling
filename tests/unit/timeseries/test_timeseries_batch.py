"""The ``TimeSeriesBatch`` specialization of the generic batch.

The window space is fixed (``("time", "channel")``, no value range) and a
single row keeps the batch's type and its ``sampling_rate``.
"""

from kcai_data_sampling_timeseries.api.selection import TimeSeriesBatch
import numpy as np
import pyarrow as pa


def test_defaults_declare_time_channel_and_no_range() -> None:
    """A batch built with no space arguments declares the window space."""
    batch = TimeSeriesBatch(
        name="s",
        dataset="d",
        ids=["a@3"],
        columns=pa.table({"series_id": ["a"]}),
        data=np.zeros((1, 4, 1)),
    )
    assert batch.sample_axes == ("time", "channel")
    assert batch.value_range is None
    assert batch.sampling_rate is None


def test_row_keeps_type_and_sampling_rate(synthetic_series_batch: TimeSeriesBatch) -> None:
    """``row(i)`` is a one-window ``TimeSeriesBatch`` aligned on every axis."""
    row = synthetic_series_batch.row(2)
    assert isinstance(row, TimeSeriesBatch)
    assert row.ids == [synthetic_series_batch.ids[2]]
    assert row.columns.to_pylist() == [synthetic_series_batch.columns.to_pylist()[2]]
    assert np.array_equal(row.data[0], synthetic_series_batch.data[2])
    assert row.sampling_rate == synthetic_series_batch.sampling_rate
    assert row.sample_axes == synthetic_series_batch.sample_axes
