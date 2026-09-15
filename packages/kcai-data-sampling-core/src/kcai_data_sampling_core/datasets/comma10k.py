"""The comma10k sample: ten MIT-licensed driving frames with per-pixel masks,
fetched into `examples/data/comma10k_sample/` by `scripts/fetch_comma10k_sample.py`."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kcai_data_sampling_core.api.selection import Sample
from kcai_data_sampling_core.utils.images import load_image

#: The mask's colour code, from the comma10k README.
CLASSES: dict[str, tuple[int, int, int]] = {
    "road": (0x40, 0x20, 0x20),
    "lane_markings": (0xFF, 0x00, 0x00),
    "undrivable": (0x80, 0x80, 0x60),
    "movable": (0x00, 0xFF, 0x66),  # vehicles, people, anything that moves
    "my_car": (0xCC, 0x00, 0xFF),
}


def load_mask(path: Path) -> np.ndarray:
    """HWC uint8 RGB, colour-coded as in `CLASSES`."""
    return np.asarray(Image.open(path).convert("RGB"))


def class_pixels(mask: np.ndarray, name: str) -> np.ndarray:
    """Boolean HW map of one class."""
    return np.all(mask == CLASSES[name], axis=-1)


def load_sample(id: str, source: dict[str, Any]) -> Sample:
    """One frame and its mask, from the two paths a selection recorded."""
    return Sample(id=id, x=load_image(source["image"]), y={"mask": load_mask(source["mask"])}, source=source)


def samples(root: Path | str, limit: int | None = None) -> list[Sample]:
    """At most `limit` frames, in name order."""
    root = Path(root)
    pngs = sorted((root / "imgs").glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no frames in {root / 'imgs'}: run scripts/fetch_comma10k_sample.py")
    return [
        load_sample(p.stem[:7], {"image": str(p), "mask": str(root / "masks" / p.name)})
        for p in pngs[:limit]
    ]


def overlay(x: np.ndarray, y: dict[str, Any], cls: str = "movable", alpha: float = 0.45) -> np.ndarray:
    """The image with one mask class tinted."""
    img = np.clip(x, 0, 1).transpose(1, 2, 0).copy()
    where = class_pixels(y["mask"], cls)
    img[where] = (1 - alpha) * img[where] + alpha * np.array(CLASSES[cls]) / 255
    return img.transpose(2, 0, 1)


def ground_truth(sample: Sample) -> str:
    """One line about the annotation, for a table."""
    return f"movable {class_pixels(sample.y['mask'], 'movable').mean():.0%}"


def describe(samples: list[Sample]) -> None:
    """Print what was found."""
    if not samples:
        print("no frames found")
        return
    print(f"{len(samples)} samples, x = {samples[0].x.shape} {samples[0].x.dtype}\n")
    for kind in sorted({k for s in samples for k in s.y}):
        present = sum(kind in s.y for s in samples)
        print(f"  {kind:8} on {present}/{len(samples)} samples")
    movable = [class_pixels(s.y["mask"], "movable").mean() for s in samples]
    print(f"\n  movable pixels per frame: {min(movable):.0%} – {max(movable):.0%}")
