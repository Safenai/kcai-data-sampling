"""Inpainting: generate a region fill — the mask math.

``build_region_mask`` is the pure function that marks the rectangle a tool
model should erase and refill; the transformation class that wraps it into the
unary contract lives in ``kcai_data_sampling_images_lama.api.transformations.inpaint``.
"""

import numpy as np


def build_region_mask(
    xs_shape: tuple[int, ...],
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    """The boolean ``(B, H, W)`` mask of the rectangle to erase.

    ``True`` marks every pixel of the ``top:top+height`` by
    ``left:left+width`` window of each sample. The window must fit inside the
    frame: the frame size is only known at run time (from the ``(H, W)`` axes
    of ``xs_shape``), so this check is apply-time, and it refuses out-of-frame
    windows loudly instead of silently wrapping.

    Args:
        xs_shape: The batch shape ``(B, H, W, C)``; the frame is read from the
            ``(H, W)`` axes.
        top: Window top offset in pixels, ``>= 0``.
        left: Window left offset in pixels, ``>= 0``.
        height: Window height in pixels, ``> 0``.
        width: Window width in pixels, ``> 0``.

    Returns:
        A boolean ``(B, H, W)`` array; ``True`` inside the window.

    Raises:
        ValueError: If ``top``/``left`` are negative or ``height``/``width``
            are not positive, or if the window leaves the frame.
    """
    if top < 0 or left < 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"inpaint region needs top/left >= 0 and height/width > 0, got {top=}, {left=}, {height=}, {width=}"
        )
    frame_h, frame_w = int(xs_shape[1]), int(xs_shape[2])
    if top + height > frame_h or left + width > frame_w:
        raise ValueError(f"inpaint region {height}x{width} at ({top},{left}) leaves the {frame_h}x{frame_w} frame")
    masks = np.zeros((int(xs_shape[0]), frame_h, frame_w), dtype=bool)
    masks[:, top : top + height, left : left + width] = True
    return masks
