"""Shared decode/encode primitives for both benchmark sides."""

from __future__ import annotations

import numpy as np

from kcai_data_sampling_job.utils.images import decode_image, encode_image


def decode_batch(manifest_rows: list[dict]) -> tuple[list[int], np.ndarray]:
    """Decode a list of manifest rows into a (B,H,W,4) uint8 array.

    Args:
        manifest_rows: Rows with at least 'path'.

    Returns:
        (ids, array) where ids is the list of row ids.
    """
    decoded: list[np.ndarray] = []
    ids: list[int] = []
    for row in manifest_rows:
        height, width, raw = decode_image(row["path"])
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
        decoded.append(arr)
        ids.append(row["id"])
    if not decoded:
        return ids, np.zeros((0, 0, 0, 4), dtype=np.uint8)
    batch = np.stack(decoded)
    return ids, batch


def encode_one(path: str, array: np.ndarray) -> None:
    """Encode a single HxWxC array as PNG.

    Args:
        path: Destination path.
        array: The pixel array.
    """
    encode_image(path, array)
