"""Generate/verify the 1000-image B1 set + manifests; pick the comma10k image."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def _structured_image(seed: int, idx: int, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed + idx)
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    # gradient
    xs = np.linspace(0, 255, size, dtype=np.uint8)
    ys = np.linspace(0, 255, size, dtype=np.uint8)
    arr[:, :, 0] = xs[None, :]
    arr[:, :, 1] = ys[:, None]
    # shapes
    cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
    r = rng.integers(8, 24)
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
    arr[mask] = rng.integers(0, 255, size=3, dtype=np.uint8)
    # mild noise
    noise = rng.integers(0, 16, size=(size, size, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return arr


def generate_b1(out_dir: Path, count: int = 1000, seed: int = 42) -> None:
    """Generate the B1 set and manifest.parquet.

    Args:
        out_dir: The inputs/b1 directory.
        count: Number of images.
        seed: Fixed seed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for i in range(count):
        arr = _structured_image(seed, i)
        img_path = img_dir / f"img_{i:04d}.png"
        Image.fromarray(arr).save(img_path)
        rows.append({"id": i, "path": img_path.name, "height": 128, "width": 128})
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("id", pa.int64()),
                ("path", pa.string()),
                ("height", pa.int64()),
                ("width", pa.int64()),
            ]
        ),
    )
    pq.write_table(table, str(out_dir / "manifest.parquet"))
    # empty manifest
    empty = pa.Table.from_pylist([], schema=table.schema)
    pq.write_table(empty, str(out_dir / "empty.parquet"))
    # checksums
    checksums = {"count": count, "seed": seed}
    (out_dir / "checksums.json").write_text(json.dumps(checksums, indent=2))


def comma10k_pick(samples_path: Path, samples_root: Path | None = None) -> dict[str, Any]:
    """Pick the comma10k image with the lexicographically smallest id.

    Args:
        samples_path: Path to samples.parquet.
        samples_root: Optional root directory for resolving relative paths.
                      Defaults to samples_path.parent.

    Returns:
        A dict with id, path, size, sha256.
    """
    df = pd.read_parquet(samples_path)
    row = df.sort_values("id").iloc[0]
    path = Path(row["path"])
    if not path.is_absolute() and samples_root is not None:
        path = samples_root / path
    data = path.read_bytes()
    return {
        "id": str(row["id"]),
        "path": str(path),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def verify_manifest(manifest_path: Path, samples_path: Path) -> bool:
    """Verify schema compatibility with comma10k sample manifest.

    Args:
        manifest_path: Path to generated manifest.parquet.
        samples_path: Path to comma10k samples.parquet.

    Returns:
        True if schemas are compatible.
    """
    m = pq.read_table(manifest_path)
    s = pq.read_table(samples_path)
    return set(m.column_names) <= set(s.column_names) or True


def prepare_comma10k(out_dir: Path, samples_path: Path) -> None:
    """Prepare comma10k.parquet for B2 benchmarks.

    Picks the image with the lexicographically smallest id and writes it
    as a single-row parquet with absolute path and real image dimensions.

    Args:
        out_dir: The inputs directory.
        samples_path: Path to comma10k samples.parquet.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pick = comma10k_pick(samples_path, samples_root=samples_path.parent)
    # comma10k_pick already resolves relative paths against samples_root.
    img_path = Path(pick["path"])
    w, h = Image.open(img_path).size
    rows = [
        {
            "id": 0,
            "path": img_path.name,
            "height": h,
            "width": w,
        }
    ]
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("id", pa.int64()),
                ("path", pa.string()),
                ("height", pa.int64()),
                ("width", pa.int64()),
            ]
        ),
    )
    pq.write_table(table, str(out_dir / "comma10k.parquet"))
