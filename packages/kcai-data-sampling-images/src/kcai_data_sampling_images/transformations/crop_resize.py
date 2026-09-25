"""Crop a central window, then resize it back to the sample shape: the math.

``crop_resize(xs, fraction, top, left)`` is the pure numpy map; the
transformation class that wraps it into the unary contract lives in
``kcai_data_sampling_images.api.transformations.crop_resize``.
"""

import numpy as np


def crop_resize(xs: np.ndarray, fraction: float, top: int = 0, left: int = 0) -> np.ndarray:
    """Keep the ``top:top+fraction*H`` by ``left:left+fraction*W`` window, resize back.

    The crop is top-left anchored; the kept window is then nearest-neighbor
    resized back to ``(H, W)`` by index-replication, so the sample space stays
    intact.

    Args:
        xs: Batch of image arrays shaped ``(B, H, W, C)``.
        fraction: Fraction ``(0, 1]`` of the image kept by the crop.
        top: Crop window top offset in pixels.
        left: Crop window left offset in pixels.

    Returns:
        A batch of the same shape, uint8.

    Raises:
        ValueError: If the crop window leaves the image (``top``/``left`` too
            large for the computed window).
    """
    height, width = xs.shape[1], xs.shape[2]
    window_h: int = max(1, int(height * fraction))
    window_w: int = max(1, int(width * fraction))
    if top + window_h > height or left + window_w > width:
        raise ValueError(
            f"crop_resize: window {window_h}x{window_w} at ({top},{left}) "
            f"leaves the {height}x{width} image"
        )
    window = xs[:, top : top + window_h, left : left + window_w, :]

    rows: np.ndarray = np.arange(height) * window_h // height
    cols: np.ndarray = np.arange(width) * window_w // width
    return window[:, rows[:, None], cols[None, :], :]