"""Scenario -> job-config dict -> YAML written under artifacts/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


from benchmark.scenarios import b2a_entries, b2b_entries, B1_METHOD


def b1_config(batch_size: int, run_dir: Path) -> dict[str, Any]:
    """Build a B1 job config dict for one cell.

    Args:
        batch_size: The cell's B.
        run_dir: The run's artifact directory.

    Returns:
        The job config dict.
    """
    return {
        "dataloaders": {
            "loaders": [
                {
                    "name": "b1",
                    "type": "parquet",
                    "path": str(run_dir / "inputs" / "manifest.parquet"),
                    "id_column": "id",
                    "load_batch_size": batch_size,
                    "sample_path": [{"column": "path", "prefix": str(run_dir / "inputs" / "images")}],
                }
            ]
        },
        "operations": {
            "outputs": {
                "path": str(run_dir / "ledger.parquet"),
                "write_samples": True,
                "samples_dir": str(run_dir / "payloads"),
                "flush_batch_size": 10000,
                "include": ["height", "width"],
            },
            "transform_batch_size": batch_size,
            "transformations": [
                {
                    "name": B1_METHOD["type"],
                    "type": B1_METHOD["type"],
                    "seed": B1_METHOD["seed"],
                    "brightness_limit": B1_METHOD["brightness_limit"],
                    "contrast_limit": B1_METHOD["contrast_limit"],
                    "ensure_safe_range": B1_METHOD["ensure_safe_range"],
                    "p": B1_METHOD["p"],
                }
            ],
        },
    }


def b2a_config(run_dir: Path) -> dict[str, Any]:
    """Build a B2-a job config dict.

    Args:
        run_dir: The run's artifact directory.

    Returns:
        The job config dict.
    """
    return {
        "dataloaders": {
            "loaders": [
                {
                    "name": "b2a",
                    "type": "parquet",
                    "path": str(run_dir / "inputs" / "comma10k.parquet"),
                    "id_column": "id",
                    "load_batch_size": 1,
                    "sample_path": [{"column": "path", "prefix": "examples/data/comma10k_sample/imgs"}],
                }
            ]
        },
        "operations": {
            "outputs": {
                "path": str(run_dir / "ledger.parquet"),
                "write_samples": True,
                "samples_dir": str(run_dir / "payloads"),
                "flush_batch_size": 10000,
                "include": ["height", "width"],
            },
            "transform_batch_size": 1,
            "transformations": b2a_entries(),
        },
    }


def b2b_config(run_dir: Path) -> dict[str, Any]:
    """Build a B2-b job config dict.

    Args:
        run_dir: The run's artifact directory.

    Returns:
        The job config dict.
    """
    return {
        "dataloaders": {
            "loaders": [
                {
                    "name": "b2b",
                    "type": "parquet",
                    "path": str(run_dir / "inputs" / "comma10k.parquet"),
                    "id_column": "id",
                    "load_batch_size": 1,
                    "sample_path": [{"column": "path", "prefix": "examples/data/comma10k_sample/imgs"}],
                }
            ]
        },
        "operations": {
            "outputs": {
                "path": str(run_dir / "ledger.parquet"),
                "write_samples": True,
                "samples_dir": str(run_dir / "payloads"),
                "flush_batch_size": 10000,
                "include": ["height", "width"],
            },
            "transform_batch_size": 1,
            "transformations": b2b_entries(),
        },
    }


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    """Write a config dict as YAML.

    Args:
        path: Destination path.
        config: The config dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
