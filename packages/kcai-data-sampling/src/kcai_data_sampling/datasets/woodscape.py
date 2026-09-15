"""WoodScape (Valeo): code MIT, data proprietary, nothing is downloaded, the
reader takes a directory a licensee filled in (`examples/data/woodscape_sample/`).

Only `rgb_images/` is required. Each recognised annotation directory is read
if present; an unrecognised one is carried by path.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kcai_data_sampling.api.selection import Sample
from kcai_data_sampling.utils.io import load_image

def read_boxes(path: Path) -> list[dict[str, Any]]:
    """`box_2d_annotations/*.txt`: `name,index,left,top,right,bottom` per
    line (a space-separated variant exists; both read the same way)."""
    boxes = []
    for line in path.read_text().splitlines():
        fields = line.replace(",", " ").split()
        if len(fields) < 6:
            continue
        boxes.append({
            "name": fields[0],
            "index": int(fields[1]),
            "box": [float(v) for v in fields[2:6]],  # left, top, right, bottom
        })
    return boxes


def read_mask(path: Path) -> np.ndarray:
    """A per-pixel annotation: semantic classes, instance ids, motion."""
    return np.asarray(Image.open(path))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


#: How to read each annotation directory; an absent one is carried by path.
READERS = {
    "box_2d_annotations": (".txt", read_boxes),
    "semantic_annotations": (".png", read_mask),
    "motion_annotations": (".png", read_mask),
    "instance_annotations": (".json", read_json),
    "calibration": (".json", read_json),
}


def files_of(directory: Path) -> Path:
    """Descend through the nested layout the Drive zips unpack to
    (`rgb_images/rgb_images/…`); a flat layout works too."""
    while (directory / directory.name).is_dir():
        directory = directory / directory.name
    return directory


def annotation_files(root: Path, stem: str) -> dict[str, str]:
    """Every annotation file this dataset holds for one stem, keyed by type."""
    found: dict[str, str] = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        name = directory.name
        if name in ("rgb_images", "previous_images"):
            continue
        suffix, _ = READERS.get(name, (None, None))
        here = files_of(directory)
        candidates = [here / f"{stem}{suffix}"] if suffix else list(here.glob(f"{stem}.*"))
        path = next((c for c in candidates if c.exists()), None)
        if path is not None:
            found[name] = str(path)
    return found


def load_sample(id: str, source: dict[str, Any]) -> Sample:
    """One image and whichever annotation files a selection recorded for it."""
    y: dict[str, Any] = {}
    for name, path in source.items():
        if name == "image":
            continue
        _, reader = READERS.get(name, (None, None))
        y[name] = reader(Path(path)) if reader else {"path": path}
    return Sample(id=id, x=load_image(source["image"]), y=y, source=source)


def samples(root: Path | str, limit: int | None = None, camera: str | None = "FV") -> list[Sample]:
    """At most `limit` images of one `camera` (`"FV"`, `"RV"`, `"MVL"`,
    `"MVR"`, or `None` for all), with whatever annotations exist for them."""
    root = Path(root)
    if not (root / "rgb_images").is_dir():
        raise FileNotFoundError(
            f"{root / 'rgb_images'} not found. WoodScape's data is proprietary and is "
            "not downloaded by this notebook, see the README in that directory."
        )

    pngs = sorted(files_of(root / "rgb_images").glob("*.png"))
    if camera:
        pngs = [p for p in pngs if p.stem.endswith(f"_{camera}")]

    return [load_sample(p.stem, {"image": str(p), **annotation_files(root, p.stem)}) for p in pngs[:limit]]


def overlay(x: np.ndarray, y: dict[str, Any], thickness: int = 3) -> np.ndarray:
    """The image with its 2D boxes drawn on."""
    img = np.clip(x, 0, 1).copy()
    h, w = img.shape[-2:]
    for entry in y.get("box_2d_annotations", []):
        l, t, r, b = (int(round(v)) for v in entry["box"])
        l, t, r, b = max(l, 0), max(t, 0), min(r, w - 1), min(b, h - 1)
        yellow = np.array([1.0, 1.0, 0.0], dtype=img.dtype)[:, None]
        for k in range(thickness):
            img[:, t + k, l:r + 1] = yellow
            img[:, max(b - k, 0), l:r + 1] = yellow
            img[:, t:b + 1, l + k] = yellow
            img[:, t:b + 1, max(r - k, 0)] = yellow
    return img


def ground_truth(sample: Sample) -> str:
    """One line about the annotation, for a table."""
    boxes = sample.y.get("box_2d_annotations", [])
    return f"{sum(e['name'] == 'vehicles' for e in boxes)} vehicles, {sum(e['name'] == 'person' for e in boxes)} persons"


def describe(samples: list[Sample]) -> None:
    """Print what was found."""
    if not samples:
        print("no images found")
        return

    print(f"{len(samples)} samples, x = {samples[0].x.shape} {samples[0].x.dtype}\n")
    for kind in sorted({k for s in samples for k in s.y}):
        present = sum(kind in s.y for s in samples)
        print(f"  {kind:24} on {present}/{len(samples)} samples")
