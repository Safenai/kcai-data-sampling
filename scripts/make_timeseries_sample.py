#!/usr/bin/env python3
"""Generate the synthetic time-series sample the example config uses, into
`examples/data/timeseries_sample/series.parquet` (git-ignored).

    python scripts/make_timeseries_sample.py [--out DIR]

Nothing is downloaded: five recordings of 2,000 steps at 100 Hz, two channels
each, built from a fixed seed so every run writes the same values.

- channel 0, "vibration": a 3 Hz sinusoid plus a weaker 20 Hz component, the
  one a low-pass at 10 Hz removes;
- channel 1, "temperature": a slow drift with a step halfway through, plus
  noise.

One row per series: `series_id`, `values` (`list<list<float>>`, 2,000 steps of
2 channels) and `machine`, a source column the windows carry along.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parents[1] / "examples" / "data" / "timeseries_sample"
SEED = 0
N_SERIES = 5
LENGTH = 2000
SAMPLING_RATE = 100.0


def make_series(seed: int = SEED) -> pa.Table:
    """Build the sample table.

    Args:
        seed: Seed of the phases, amplitudes and noise.

    Returns:
        One row per series: ``series_id``, ``values`` and ``machine``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(LENGTH) / SAMPLING_RATE
    rows = []
    for _ in range(N_SERIES):
        vibration = np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 2 * np.pi))
        vibration += rng.uniform(0.2, 0.5) * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 2 * np.pi))
        temperature = 20.0 + 0.002 * np.arange(LENGTH) + 2.0 * (np.arange(LENGTH) >= LENGTH // 2)
        temperature += rng.normal(scale=0.1, size=LENGTH)
        rows.append(np.stack([vibration, temperature], axis=1).tolist())
    return pa.table(
        {
            "series_id": [f"rec_{i:02d}" for i in range(N_SERIES)],
            "values": rows,
            "machine": [f"m{i % 2}" for i in range(N_SERIES)],
        }
    )


def main() -> None:
    """Write ``series.parquet`` into ``--out`` (default: the examples folder)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "series.parquet"
    pq.write_table(make_series(), path)
    print(f"wrote {N_SERIES} series of {LENGTH} steps at {SAMPLING_RATE:g} Hz to {path}")


if __name__ == "__main__":
    main()
