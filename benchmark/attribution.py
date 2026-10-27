"""Attribution: M3 (cProfile) and M5 (in-memory microbench)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cProfile
import io
import pstats


def run_m3_profiler(func, *args, **kwargs) -> dict[str, Any]:
    """Profile a function with cProfile and return cumulative stats.

    Args:
        func: The function to profile.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        A dict with total time and top functions.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = func(*args, **kwargs)
    finally:
        profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(10)
    return {"result": result, "profile_text": stream.getvalue()}


def run_m5_microbench(func, iterations: int = 100, warmup: int = 5) -> dict[str, Any]:
    """Microbenchmark a function over K iterations.

    Each iteration is timed separately so the report gets a median and an IQR
    rather than a mean that a single scheduler hiccup can move.

    Args:
        func: The function to benchmark.
        iterations: Number of timed iterations.
        warmup: Number of untimed iterations before timing.

    Returns:
        A dict with iterations, warmup, median ms per call, IQR and mean.
    """
    import statistics
    import time

    for _ in range(warmup):
        func()

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        samples.append((time.perf_counter() - start) * 1000.0)

    median = statistics.median(samples)
    quartiles = statistics.quantiles(samples, n=4) if len(samples) > 1 else [median, median, median]
    return {
        "iterations": iterations,
        "warmup": warmup,
        "ms_per_call": median,
        "ms_iqr": quartiles[2] - quartiles[0],
        "ms_mean": statistics.fmean(samples),
    }


def run_m3_full_job(config_path: Path) -> dict[str, Any]:
    """Profile the full job (ours side) by calling kcai-data-sampling execute.

    Args:
        config_path: Path to the YAML config for the job.

    Returns:
        A dict with the profile text.
    """
    from kcai_data_sampling.__main__ import execute

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        execute(["process", "--process-config", str(config_path)])
    finally:
        profiler.disable()
    stats = pstats.Stats(profiler)

    # Extract batch-scaling functions: ncalls not in {1,2,1000}, tottime >= 0.0005
    batch_scaling: list[dict[str, Any]] = []
    for key, (_cc, nc, tt, ct, _callers) in stats.stats.items():
        filename, lineno, funcname = key
        if nc <= 2:
            continue
        if tt < 0.0005:
            continue
        batch_scaling.append({
            "function": f"{filename.split('/')[-1]}:{lineno}({funcname})",
            "ncalls": nc,
            "tottime": round(tt, 6),
            "cumtime": round(ct, 6),
            "per_call_ms": round(tt / nc * 1000, 4),
        })
    batch_scaling.sort(key=lambda x: -x["tottime"])

    # Also produce a condensed text (top 20 by cumtime for context)
    stream = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream)
    stats2.sort_stats("cumulative")
    stats2.print_stats(20)
    return {
        "profile_text": stream.getvalue(),
        "batch_scaling": batch_scaling,
    }


def write_attribution(path: Path, data: dict[str, Any]) -> None:
    """Write attribution.json.

    Args:
        path: Destination path.
        data: The attribution data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
