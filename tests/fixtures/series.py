"""Synthetic meaningful time-series test data.

Like the image fixtures, every window here is **decodable by eye**: each one is
a sum of sinusoids whose frequencies sit exactly on FFT bins of the window, so
a filter that keeps or removes a component is checked to float precision, not
by a loose tolerance. A seeded phase and offset make the exact values depend
on ``KCAI_TEST_SEED``.
"""

from __future__ import annotations

from pathlib import Path

from kcai_data_sampling_job.dataloaders.api.timeseries import window_id
from kcai_data_sampling_timeseries.api.selection import TimeSeriesBatch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SERIES_WINDOWS = 6
SERIES_WIDTH = 200
SERIES_CHANNELS = 2
SERIES_SELECTION = "synthetic_series"
SERIES_DATASET = "parquet_series_synthetic"

#: Cycles per step of the two components: 10 and 60 full periods over a
#: 200-step window, so each falls exactly on one FFT bin.
LOW_FREQUENCY = 0.05
HIGH_FREQUENCY = 0.3


def make_windows(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(B, W, C)`` windows and their low-frequency part alone.

    Each channel is ``offset + sin(low) + 0.5 * sin(high)``, with a seeded
    phase per component and a seeded offset per window and channel.

    Args:
        seed: The suite seed.

    Returns:
        ``(windows, low_part)``: the float64 windows, and the same windows
        without the high-frequency component (what an ideal low-pass between
        the two frequencies must return).
    """
    rng = np.random.default_rng(seed)
    steps = np.arange(SERIES_WIDTH)[None, :, None]
    shape = (SERIES_WINDOWS, 1, SERIES_CHANNELS)
    offset = rng.normal(size=shape)
    low_phase = rng.uniform(0, 2 * np.pi, size=shape)
    high_phase = rng.uniform(0, 2 * np.pi, size=shape)
    low_part = offset + np.sin(2 * np.pi * LOW_FREQUENCY * steps + low_phase)
    high_part = 0.5 * np.sin(2 * np.pi * HIGH_FREQUENCY * steps + high_phase)
    return low_part + high_part, low_part


@pytest.fixture(scope="session")
def synthetic_series_batch(test_seed: int) -> TimeSeriesBatch:
    """A ``TimeSeriesBatch`` of synthetic windows, as a loader would emit.

    Returns:
        The 6-window batch for the current seed, at 100 Hz.
    """
    windows, _ = make_windows(test_seed)
    n = len(windows)
    return TimeSeriesBatch(
        name=SERIES_SELECTION,
        dataset=SERIES_DATASET,
        ids=[window_id(f"series_{i}", 0, SERIES_WIDTH - 1) for i in range(n)],
        columns=pa.table({"series_id": [f"series_{i}" for i in range(n)], "start": [0] * n}),
        data=windows,
        sampling_rate=100.0,
    )


#: The series of the parquet fixture: id → number of steps. ``short`` holds
#: no 32-step window, so it yields nothing and is not a failure.
SERIES_LENGTHS = {"s0": 100, "s1": 70, "s2": 64, "short": 20}


@pytest.fixture(scope="session")
def series_parquet(test_seed: int, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A parquet table of four 2-channel series, one row per series.

    Columns: ``series_id``, ``values`` (``list<list<float>>``, seeded noise
    around a slow sinusoid) and ``site``, a source column for pass-through.

    Returns:
        The parquet file path.
    """
    rng = np.random.default_rng(test_seed)
    values = []
    for n in SERIES_LENGTHS.values():
        slow = np.sin(2 * np.pi * 0.02 * np.arange(n))[:, None]
        values.append((slow + 0.3 * rng.normal(size=(n, 2))).tolist())
    table = pa.table(
        {
            "series_id": list(SERIES_LENGTHS),
            "values": values,
            "site": [f"site_{i % 2}" for i in range(len(SERIES_LENGTHS))],
        }
    )
    path = tmp_path_factory.mktemp("series") / "series.parquet"
    pq.write_table(table, path)
    return path
