"""Phase 1 timing analyzer."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any


def median(values: list[float]) -> float:
    """Compute the median of a list of values.

    Args:
        values: List of numeric values.

    Returns:
        The median.
    """
    if not values:
        return 0.0
    return statistics.median(values)


def iqr(values: list[float]) -> float:
    """Compute the interquartile range.

    Args:
        values: List of numeric values.

    Returns:
        IQR (Q3 - Q1).
    """
    if len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[3 * n // 4]
    return q3 - q1


def aggregate_timing(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate raw run records into per-cell timing summaries.

    Args:
        records: List of raw run records.

    Returns:
        List of summary dicts.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        key = (r.get("cell", ""), r.get("side", ""))
        groups.setdefault(key, []).append(r)

    summaries = []
    for (cell, side), group in groups.items():
        walls = [g["wall_s"] for g in group if not g.get("warmup", False)]
        summaries.append(
            {
                "cell": cell,
                "side": side,
                "n_timed": len(walls),
                "median_s": median(walls),
                "iqr_s": iqr(walls),
                "min_s": min(walls) if walls else 0.0,
                "max_s": max(walls) if walls else 0.0,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to a CSV file.

    Args:
        path: Destination path.
        rows: List of row dicts.
        fieldnames: Column names.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
