"""Images on disk. Memory is CHW float32 in [0, 1]; disk is PNG."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path | str) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("RGB"), dtype="float32") / 255.0
    return array.transpose(2, 0, 1)


def quantize(x: np.ndarray) -> np.ndarray:
    """The HWC uint8 array a PNG will hold."""
    return (np.clip(x, 0, 1).transpose(1, 2, 0) * 255).round().astype("uint8")


def save_image(path: Path | str, x: np.ndarray) -> None:
    Image.fromarray(quantize(x)).save(path)
