"""Phase 2 memory analyzer."""

from __future__ import annotations

from typing import Any


def aggregate_memory(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate raw run records into per-cell memory summaries.

    Args:
        records: List of raw run records.

    Returns:
        List of summary dicts with peak RSS stats.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        key = (r.get("cell", ""), r.get("side", ""))
        groups.setdefault(key, []).append(r)

    summaries = []
    for (cell, side), group in groups.items():
        peaks = [g.get("peak_rss_kb", 0) / 1024.0 for g in group if not g.get("warmup", False)]
        if not peaks:
            continue
        summaries.append(
            {
                "cell": cell,
                "side": side,
                "n_timed": len(peaks),
                "peak_rss_mib_median": _median(peaks),
                "peak_rss_mib_iqr": _iqr(peaks),
            }
        )
    return summaries


def _median(values: list[float]) -> float:
    import statistics

    return statistics.median(values) if values else 0.0


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return sorted_vals[3 * n // 4] - sorted_vals[n // 4]
