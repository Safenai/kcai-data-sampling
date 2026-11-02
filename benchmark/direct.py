"""Side D: matched baseline harness using albumentations directly."""

from __future__ import annotations

import hashlib
import numpy as np
from pathlib import Path
from typing import Any

import albumentations
import pyarrow as pa
import pyarrow.parquet as pq

from benchmark.parity import decode_batch, encode_one
from benchmark.scenarios import (
    S1_METHOD,
    s2_entries,
    s3_entries,
)


def run_direct(suite: str, config_path: Path, artifact_dir: Path) -> None:
    """Run the baseline side for one suite.

    Args:
        suite: The suite name (s1, s2, s3).
        config_path: Path to the job config YAML.
        artifact_dir: Directory for artifacts.
    """
    import yaml

    with config_path.open() as f:
        config = yaml.safe_load(f)

    loader = config["dataloaders"]["loaders"][0]
    import pandas as pd

    manifest = pd.read_parquet(loader["path"])
    rows = manifest.to_dict("records")

    # Resolve image paths using sample_path prefix
    sample_path_cfg = (loader.get("sample_path") or [{}])[0]
    prefix = sample_path_cfg.get("prefix", "")
    img_column = sample_path_cfg.get("column", "path")
    if prefix:
        for row in rows:
            row[img_column] = str(Path(prefix) / row[img_column])

    ledger_rows: list[dict[str, Any]] = []
    payloads_dir = artifact_dir / "payloads"
    payloads_dir.mkdir(parents=True, exist_ok=True)

    if suite == "s1":
        _run_s1(rows, config, payloads_dir, ledger_rows)
    elif suite == "s2":
        _run_s2(rows, payloads_dir, ledger_rows)
    elif suite == "s3":
        _run_s3(rows, payloads_dir, ledger_rows)

    # Write minimal parity parquet
    table = pa.Table.from_pylist(
        ledger_rows,
        schema=pa.schema(
            [
                ("id", pa.int64()),
                ("artifact", pa.string()),
                ("sha256", pa.string()),
            ]
        ),
    )
    pq.write_table(table, str(artifact_dir / "ledger.parquet"))


def _run_s1(
    rows: list[dict],
    config: dict,
    payloads_dir: Path,
    ledger_rows: list[dict],
) -> None:
    """Run S1: random_brightness_contrast at each batch size.

    Args:
        rows: Manifest rows.
        config: Job config dict.
        payloads_dir: Directory for output PNGs.
        ledger_rows: Accumulator for ledger rows.
    """
    bs = config.get("compute", {}).get("load_batch_size", 1)
    for i in range(0, len(rows), bs):
        chunk = rows[i : i + bs]
        ids, batch = decode_batch(chunk)
        # Create transform with fixed seed
        t = albumentations.RandomBrightnessContrast(
            brightness_limit=S1_METHOD["brightness_limit"],
            contrast_limit=S1_METHOD["contrast_limit"],
            p=S1_METHOD["p"],
        )
        # Apply per row
        for j, row_id in enumerate(ids):
            img = batch[j]
            result = t(image=img)
            out = result["image"]
            artifact = f"b1__{row_id}__{hashlib.sha1(out.tobytes()).hexdigest()[:6]}.png"
            encode_one(str(payloads_dir / artifact), out)
            sha = hashlib.sha256((payloads_dir / artifact).read_bytes()).hexdigest()
            ledger_rows.append({"id": row_id, "artifact": artifact, "sha256": sha})


def _run_s2(
    rows: list[dict],
    payloads_dir: Path,
    ledger_rows: list[dict],
) -> None:
    """Run S2: 40 transform types, one per image.

    Args:
        rows: Manifest rows (single image for S2/S3).
        payloads_dir: Directory for output PNGs.
        ledger_rows: Accumulator for ledger rows.
    """
    # Decode each row once (not once per transform entry)
    decoded: dict[Any, np.ndarray] = {}
    for row in rows:
        _, batch = decode_batch([row])
        decoded[row["id"]] = batch[0]

    for entry in s2_entries():
        for row in rows:
            img = decoded[row["id"]]
            # Build transform from entry
            t = _build_transform(entry)
            if isinstance(t, albumentations.ImageOnlyTransform) and img.shape[-1] == 4:
                rgb, alpha = img[..., :3], img[..., 3:]
                transformed_rgb = t(image=rgb)["image"]
                out = np.concatenate([transformed_rgb, alpha], axis=-1)
            else:
                result = t(image=img)
                out = result["image"]
            artifact = f"s2__{row['id']}__{entry['name']}__{hashlib.sha1(out.tobytes()).hexdigest()[:6]}.png"
            encode_one(str(payloads_dir / artifact), out)
            sha = hashlib.sha256((payloads_dir / artifact).read_bytes()).hexdigest()
            ledger_rows.append({"id": row["id"], "artifact": artifact, "sha256": sha})


def _run_s3(
    rows: list[dict],
    payloads_dir: Path,
    ledger_rows: list[dict],
) -> None:
    """Run S3: gaussian_blur at 40 sigma levels.

    Args:
        rows: Manifest rows (single image for S2/S3).
        payloads_dir: Directory for output PNGs.
        ledger_rows: Accumulator for ledger rows.
    """
    # Decode each row once (not once per transform entry)
    decoded: dict[Any, np.ndarray] = {}
    for row in rows:
        _, batch = decode_batch([row])
        decoded[row["id"]] = batch[0]

    for entry in s3_entries():
        sigma = entry["sigma_limit"][0]
        for row in rows:
            img = decoded[row["id"]]
            t = albumentations.GaussianBlur(sigma_limit=(sigma, sigma), p=1.0)
            if isinstance(t, albumentations.ImageOnlyTransform) and img.shape[-1] == 4:
                rgb, alpha = img[..., :3], img[..., 3:]
                transformed_rgb = t(image=rgb)["image"]
                out = np.concatenate([transformed_rgb, alpha], axis=-1)
            else:
                result = t(image=img)
                out = result["image"]
            artifact = f"s3__{row['id']}__{entry['name']}__{hashlib.sha1(out.tobytes()).hexdigest()[:6]}.png"
            encode_one(str(payloads_dir / artifact), out)
            sha = hashlib.sha256((payloads_dir / artifact).read_bytes()).hexdigest()
            ledger_rows.append({"id": row["id"], "artifact": artifact, "sha256": sha})


def _build_transform(entry: dict) -> albumentations.BasicTransform:
    """Build an albumentations transform from a scenario entry.

    Args:
        entry: Dict with 'type' and 'params'.

    Returns:
        An albumentations transform instance.
    """
    cls_name = entry["type"]
    # Convert snake_case to CamelCase
    cls_name = "".join(word.capitalize() for word in cls_name.split("_"))

    # Special cases where the CamelCase doesn't match the actual class name
    special_cases = {
        "Isonoise": "ISONoise",
        "Iaaadditivegaussiannoise": "IAAAdditiveGaussianNoise",
        "Clahe": "CLAHE",
        "RgbShift": "RGBShift",
    }
    cls_name = special_cases.get(cls_name, cls_name)

    params = entry.get("params", {})
    cls = getattr(albumentations, cls_name)
    # Filter params to match constructor signature
    import inspect

    sig = inspect.signature(cls.__init__)
    valid_params = {k: v for k, v in params.items() if k in sig.parameters}

    # Add defaults for required parameters that transforms may need
    # Some transforms like Downscale need height/width for validation
    if "height" in sig.parameters and "height" not in valid_params:
        valid_params["height"] = 128
    if "width" in sig.parameters and "width" not in valid_params:
        valid_params["width"] = 128

    return cls(**valid_params, p=1.0)
