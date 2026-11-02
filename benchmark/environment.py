"""Environment probe and digest for reproducible benchmark runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("numpy", "cv2", "pyarrow", "PIL", "albucore", "albumentations"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = "?"
    return out


def _albumentations_direct_url() -> str | None:
    try:
        import importlib.metadata as md

        dist = md.distribution("kcai-data-sampling-albumentation")
        raw = dist.read_text("direct_url.json")
        if raw:
            import json as _json

            data = _json.loads(raw)
            return data.get("url") or data.get("vcs_info", {}).get("commit_id")
    except Exception:
        return None
    return None


def probe() -> dict[str, Any]:
    """Build the environment receipt.

    Returns:
        A dict with run metadata, versions, and thread info.
    """
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "nproc": os.cpu_count() or 1,
        "versions": _versions(),
        "albumentations_direct_url": _albumentations_direct_url(),
        "env": {
            "NO_ALBUMENTATIONS_UPDATE": os.environ.get("NO_ALBUMENTATIONS_UPDATE"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "cv2_num_threads": _cv2_threads(),
        },
    }


def _cv2_threads() -> int | None:
    try:
        import cv2

        return int(cv2.getNumThreads())
    except Exception:
        return None


def write_environment(path: Path, receipt: dict[str, Any]) -> None:
    """Write the environment receipt to a JSON file.

    Args:
        path: Destination path.
        receipt: The environment receipt dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)


def git_sha() -> str:
    """Return the short git SHA of HEAD, or 'unknown' if unavailable.

    Returns:
        The short SHA string.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """Check if the working tree has uncommitted changes.

    Returns:
        True if dirty, False otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False
