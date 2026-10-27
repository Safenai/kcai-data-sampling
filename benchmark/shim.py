"""Subprocess shim: runs side A or D and writes telemetry.json at exit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

from kcai_data_sampling.__main__ import execute as run_job


def run_side_ours(config_path: Path, telemetry_path: Path, artifact_dir: Path) -> int:
    """Run side A (ours) using the job CLI.

    Args:
        config_path: Path to the YAML config.
        telemetry_path: Path to write telemetry.json.
        artifact_dir: Directory for artifacts.

    Returns:
        Exit code.
    """
    import shutil

    # Derive the run-level directory (outputs go here per config)
    run_level = artifact_dir.parents[4]

    # Clean previous outputs at run level
    ledger_path = run_level / "ledger.parquet"
    payloads_path = run_level / "payloads"
    if ledger_path.exists():
        ledger_path.unlink()
    if payloads_path.exists():
        shutil.rmtree(payloads_path)
    payloads_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        run_job(["process", "--process-config", str(config_path)])
        exit_code = 0
    except Exception as exc:
        print(f"side ours failed: {exc}", file=sys.stderr)
        exit_code = 1
    elapsed = time.perf_counter() - start

    # Copy outputs to per-run artifact directory
    if exit_code == 0:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if ledger_path.exists():
            shutil.copy2(ledger_path, artifact_dir / "ledger.parquet")
        if payloads_path.exists():
            dest_payloads = artifact_dir / "payloads"
            if dest_payloads.exists():
                shutil.rmtree(dest_payloads)
            shutil.copytree(payloads_path, dest_payloads)

    _write_telemetry(telemetry_path, elapsed, exit_code, artifact_dir)
    return exit_code


def run_side_direct(suite: str, config_path: Path, telemetry_path: Path, artifact_dir: Path) -> int:
    """Run side D (baseline) via direct.py.

    Args:
        suite: The suite name (s1, s2, s3).
        config_path: Path to the YAML config.
        telemetry_path: Path to write telemetry.json.
        artifact_dir: Directory for artifacts.

    Returns:
        Exit code.
    """
    start = time.perf_counter()
    try:
        from benchmark.direct import run_direct

        run_direct(suite, config_path, artifact_dir)
        exit_code = 0
    except Exception as exc:
        print(f"side direct failed: {exc}", file=sys.stderr)
        exit_code = 1
    elapsed = time.perf_counter() - start
    _write_telemetry(telemetry_path, elapsed, exit_code, artifact_dir)
    return exit_code


def _write_telemetry(path: Path, wall_s: float, exit_code: int, artifact_dir: Path) -> None:
    """Write telemetry.json with peak RSS and timing.

    Args:
        path: Destination path.
        wall_s: Wall-clock seconds.
        exit_code: Process exit code.
        artifact_dir: Artifact directory for row/payload counts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_kb = usage.ru_maxrss
    rows = _count_rows(artifact_dir)
    payloads = _count_payloads(artifact_dir)
    with path.open("w") as f:
        json.dump(
            {
                "wall_s": wall_s,
                "peak_rss_kb": peak_kb,
                "rows": rows,
                "payloads": payloads,
                "exit_code": exit_code,
            },
            f,
        )


def _count_rows(artifact_dir: Path) -> int:
    """Count ledger rows from the parquet file if present.

    Args:
        artifact_dir: The run's artifact directory.

    Returns:
        Number of rows, or 0 if not found.
    """
    ledger = artifact_dir / "ledger.parquet"
    if not ledger.exists():
        return 0
    try:
        import pyarrow.parquet as pq

        return pq.read_table(ledger).num_rows
    except Exception:
        return 0


def _count_payloads(artifact_dir: Path) -> int:
    """Count payload PNGs in the payloads directory.

    Args:
        artifact_dir: The run's artifact directory.

    Returns:
        Number of PNG files.
    """
    payloads_dir = artifact_dir / "payloads"
    if not payloads_dir.exists():
        return 0
    return len(list(payloads_dir.glob("*.png")))


def main(argv: list[str] | None = None) -> int:
    """Entry point for the shim.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(prog="shim")
    parser.add_argument("--side", choices=["ours", "direct"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", choices=["s1", "s2", "s3"], default="s1")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.side == "ours":
        return run_side_ours(args.config, args.telemetry, args.artifacts)
    return run_side_direct(args.suite, args.config, args.telemetry, args.artifacts)


if __name__ == "__main__":
    raise SystemExit(main())
