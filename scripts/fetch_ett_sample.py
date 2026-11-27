#!/usr/bin/env python3
"""Download the ETT (Electricity Transformer Temperature) series and write them
as a `timeseries_parquet` sample, into `examples/data/ett_sample/` (git-ignored).

    python scripts/fetch_ett_sample.py [--variants ETTh1 ETTh2] [--out DIR]

ETT: https://github.com/zhouhaoyi/ETDataset (Zhou et al., Informer, AAAI 2021),
licence CC BY-ND 4.0. Nothing derived from it is committed: each user downloads
it with this script.

One row per transformer file: `series_id` (`ETTh1`, …), `values`
(`list<fixed_size_list<double>[7]>`, one step per row of the CSV, 7 channels in
the CSV order `HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`, so the oil temperature
`OT` is channel 6), `n_steps` (row count, so window discovery does not reread
`values`), `start` (timestamp of the first step) and `freq` (`1h` or `15min`),
source columns the windows can carry along. The parquet is written with zstd
compression and column statistics.
"""

import argparse
import csv
import io
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
RAW = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{COMMIT}/ETT-small"
CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
FREQS = {"ETTh1": "1h", "ETTh2": "1h", "ETTm1": "15min", "ETTm2": "15min"}
STEPS = {"1h": timedelta(hours=1), "15min": timedelta(minutes=15)}
VALUES_TYPE = pa.list_(pa.list_(pa.float64(), len(CHANNELS)))
SCHEMA = pa.schema(
    [
        pa.field("series_id", pa.string()),
        pa.field("values", VALUES_TYPE),
        pa.field("n_steps", pa.int64()),
        pa.field("start", pa.string()),
        pa.field("freq", pa.string()),
    ]
)
OUT = Path(__file__).resolve().parents[1] / "examples" / "data" / "ett_sample"


def fetch_csv(variant: str) -> str:
    """Download one ETT CSV at the pinned commit.

    Args:
        variant: File name without extension, e.g. ``ETTh1``.

    Returns:
        The CSV text.
    """
    with urllib.request.urlopen(f"{RAW}/{variant}.csv", timeout=120) as r:
        return r.read().decode("utf-8")


def parse_series(variant: str, text: str) -> tuple[str, list[list[float]]]:
    """Parse one ETT CSV into its start timestamp and per-step channel values.

    Args:
        variant: File name without extension, used in error messages.
        text: The CSV text, header ``date`` followed by the seven channels.

    Returns:
        The first timestamp (ISO format) and one ``[7 floats]`` list per step.

    Raises:
        ValueError: If the header is not the expected one, or the timestamps
            are not evenly spaced at the variant's frequency (the loader
            assumes one row per step, with no gap).
    """
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    if tuple(header) != ("date", *CHANNELS):
        raise ValueError(f"{variant}: unexpected header {header}")
    step = STEPS[FREQS[variant]]
    dates = [datetime.fromisoformat(r[0]) for r in body]
    gaps = [i for i in range(1, len(dates)) if dates[i] - dates[i - 1] != step]
    if gaps:
        raise ValueError(f"{variant}: {len(gaps)} uneven steps, first after {dates[gaps[0] - 1]}")
    return dates[0].isoformat(sep=" "), [[float(v) for v in r[1:]] for r in body]


def main() -> None:
    """Fetch the requested variants and write ``series.parquet`` into ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", default=["ETTh1", "ETTh2"], choices=sorted(FREQS))
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory.")
    args = parser.parse_args()
    ids, values, starts = [], [], []
    for variant in args.variants:
        start, series = parse_series(variant, fetch_csv(variant))
        ids.append(variant)
        values.append(series)
        starts.append(start)
        print(f"  {variant}: {len(series)} steps of {len(CHANNELS)} channels from {start}")
    args.out.mkdir(parents=True, exist_ok=True)
    gitignore = args.out / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n")
    table = pa.table(
        {
            "series_id": pa.array(ids, pa.string()),
            "values": pa.array(values, VALUES_TYPE),
            "n_steps": pa.array([len(series) for series in values], pa.int64()),
            "start": pa.array(starts, pa.string()),
            "freq": pa.array([FREQS[v] for v in args.variants], pa.string()),
        },
        schema=SCHEMA,
    )
    path = args.out / "series.parquet"
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    print(f"wrote {len(ids)} series to {path} (channels {', '.join(CHANNELS)}; OT = channel 6)")
    print("ETT is licensed CC BY-ND 4.0: do not commit or share files derived from it.")


if __name__ == "__main__":
    main()
