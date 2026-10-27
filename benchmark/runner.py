"""Subprocess runner: spawn, wall-clock, repeats, interleaving, fresh dirs, raw log."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


def _suite_for_cell(cell: str) -> str:
    """Map a cell name to the suite the direct shim should run.

    Args:
        cell: Cell name (e.g. 'b1_b5', 'b2a').

    Returns:
        The suite name: 'b1', 'b2a' or 'b2b'.
    """
    if cell.startswith("b1") or cell in ("b5", "b10", "b20"):
        return "b1"
    if cell == "b2a":
        return "b2a"
    if cell == "b2b":
        return "b2b"
    return "b1"


def _read_vmhwm_kb(pid: int) -> int:
    """Read one process's peak resident set size from /proc.

    Args:
        pid: Process id.

    Returns:
        VmHWM in KiB, or 0 when the sample cannot be read.
    """
    try:
        with Path(f"/proc/{pid}/status").open() as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError, OSError):
        return 0
    return 0


def _poll_vmhwm(pid: int, done: threading.Event, samples: list[int]) -> None:
    """Sample VmHWM every 50 ms until the child exits.

    Args:
        pid: Child process id.
        done: Event set once the child has exited.
        samples: Accumulator for the sampled VmHWM values.
    """
    while not done.is_set():
        samples.append(_read_vmhwm_kb(pid))
        time.sleep(0.05)


def _spawn(cmd: list[str]) -> tuple[subprocess.CompletedProcess[str], int]:
    """Run one subprocess while sampling its VmHWM.

    Args:
        cmd: Command line.

    Returns:
        The completed process and the peak VmHWM in KiB (0 when unavailable).
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    done = threading.Event()
    samples: list[int] = []
    poller = threading.Thread(target=_poll_vmhwm, args=(proc.pid, done, samples), daemon=True)
    poller.start()
    stdout, stderr = proc.communicate()
    done.set()
    poller.join(timeout=1.0)
    result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    return result, max(samples, default=0)


def run_cell(
    cell: str, side: str, repeat: int, config_path: Path, artifact_dir: Path, telemetry_path: Path
) -> dict[str, Any]:
    """Run one cell/side/repeat as a subprocess.

    Args:
        cell: Cell name.
        side: 'ours' or 'direct'.
        repeat: Repeat index.
        config_path: Path to config YAML.
        artifact_dir: Artifact directory.
        telemetry_path: Telemetry output path.

    Returns:
        A dict with wall_s, telemetry, exit_code.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "benchmark.shim",
        "--side",
        side,
        "--suite",
        _suite_for_cell(cell),
        "--config",
        str(config_path),
        "--artifacts",
        str(artifact_dir),
        "--telemetry",
        str(telemetry_path),
    ]

    start = time.perf_counter()
    result, vmhwm_peak_kb = _spawn(cmd)
    wall = time.perf_counter() - start

    telemetry = json.loads(telemetry_path.read_text()) if telemetry_path.exists() else {}
    if vmhwm_peak_kb > 0:
        telemetry["vmhwm_peak_kb"] = vmhwm_peak_kb

    return {
        "cell": cell,
        "side": side,
        "repeat": repeat,
        "wall_s": wall,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        **telemetry,
    }


def append_raw_log(path: Path, record: dict[str, Any]) -> None:
    """Append a record to raw_runs.jsonl.

    Args:
        path: Path to the JSONL file.
        record: The record to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
