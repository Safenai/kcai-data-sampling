#!/usr/bin/env python3
"""Download the two toy series of the newcomer demo, Air Passengers and Nile,
and write them as `timeseries_parquet` samples, into `examples/data/toy_sample/`
(git-ignored).

    python scripts/fetch_toy_samples.py [--out DIR]

- Air Passengers: monthly international airline passengers, 1949-1960, in
  thousands (Box, Jenkins & Reinsel 1976, "Series G"; R `datasets`).
- Nile: yearly flow of the Nile at Aswan, 1871-1970, in 10^8 m^3 (Cobb 1978;
  R `datasets`), with a level shift around 1899.

Source: the Rdatasets CSVs at a pinned commit
(https://github.com/vincentarelbundock/Rdatasets). Rdatasets could not
establish the data licence for certain and believes the datasets are free to
redistribute; statsmodels marks Nile as public domain.

One file per series (`air_passengers.parquet`, `nile.parquet`), one row each:
`series_id`, `values` (`list<list<float>>`, one channel, the raw values),
`log_values` (the same steps, natural logarithm: on that scale an offset is a
relative change), `start` (first period) and `freq` (`1M` or `1Y`). A config
picks the scale with the loader's `values_column`.
"""

import argparse
import csv
import io
import math
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

COMMIT = "1dcc2bf5f955cc1224a3e1307256e1fe86b68dae"
RAW = f"https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/{COMMIT}/csv/datasets"
SERIES = {
    # file name: (Rdatasets name, first period, frequency, expected length)
    "air_passengers": ("AirPassengers", "1949-01", "1M", 144),
    "nile": ("Nile", "1871", "1Y", 100),
}
OUT = Path(__file__).resolve().parents[1] / "examples" / "data" / "toy_sample"


def fetch_values(name: str, length: int) -> list[float]:
    """Download one Rdatasets CSV and return its ``value`` column.

    Args:
        name: Rdatasets name, e.g. ``AirPassengers``.
        length: Number of rows expected.

    Returns:
        The values, in time order.

    Raises:
        ValueError: If the CSV has another header or length than expected.
    """
    with urllib.request.urlopen(f"{RAW}/{name}.csv", timeout=120) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
    if not rows or set(rows[0]) != {"rownames", "time", "value"}:
        raise ValueError(f"{name}: unexpected columns {list(rows[0]) if rows else []}")
    if len(rows) != length:
        raise ValueError(f"{name}: {len(rows)} rows, expected {length}")
    return [float(row["value"]) for row in rows]


def main() -> None:
    """Fetch both series and write one parquet file each into ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gitignore = args.out / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n")
    for file_name, (name, start, freq, length) in SERIES.items():
        values = fetch_values(name, length)
        table = pa.table(
            {
                "series_id": [file_name],
                "values": [[[v] for v in values]],
                "log_values": [[[math.log(v)] for v in values]],
                "start": [start],
                "freq": [freq],
            }
        )
        path = args.out / f"{file_name}.parquet"
        pq.write_table(table, path)
        print(f"wrote {path}: {length} steps from {start} ({freq}), min {min(values):g}, max {max(values):g}")


if __name__ == "__main__":
    main()
