"""Core + images smoke: the image algorithms on an ImageBatch, no -job.

Runs inside a ``core+images`` venv: flips and crops a synthetic ``ImageBatch``
through the ``api.transformations`` classes and asserts the sample space is
preserved on the wheel-installed package.
"""

import sys

from kcai_data_sampling_images.api.selection import ImageBatch
from kcai_data_sampling_images.api.transformations.crop_resize import CropResize
from kcai_data_sampling_images.api.transformations.horizontal_flip import HorizontalFlip
import numpy as np


def _frame() -> np.ndarray:
    """A small two-tone RGBA frame (left bright, right dark)."""
    out = np.zeros((8, 8, 4), dtype=np.uint8)
    out[:, :4, :3] = 220
    out[:, 4:, :3] = 30
    out[:, :, 3] = 255
    return out


def main() -> int:
    frames = np.stack([_frame() for _ in range(4)])
    batch = ImageBatch(
        name="smoke",
        dataset="smoke",
        ids=[f"i{i}" for i in range(4)],
        columns=None,
        data=frames,
    )
    flipped = HorizontalFlip({}).transform(batch, batch.value_range)
    cropped = CropResize({"fraction": 0.5}).transform(batch, batch.value_range)
    assert all(o.x.shape == (8, 8, 4) for o in flipped)
    assert all(o.x.shape == (8, 8, 4) for o in cropped)
    assert all(o.x.dtype == np.uint8 for o in flipped)
    assert all(o.x.min() >= 0 and o.x.max() <= 255 for o in flipped)
    # The flip swaps the two-tone halves: the left color must land on the right.
    assert np.array_equal(flipped[0].x[:, -1, :], batch.data[0][:, 0, :])
    print("smoke_images ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
