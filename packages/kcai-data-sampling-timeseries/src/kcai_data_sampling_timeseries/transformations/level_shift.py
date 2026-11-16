"""Shift the level from a step on: the algorithm math.

``level_shift(xs, offset, start)`` is the pure numpy map; the transformation
class that wraps it into the unary contract lives in
``kcai_data_sampling_timeseries.api.transformations.level_shift``.
"""

import math

import numpy as np

from kcai_data_sampling_timeseries.transformations._checks import require_windows


def shift_step(length: int, start: float) -> int:
    """First step of a window of ``length`` steps that the shift reaches.

    Args:
        length: Number of steps of the window.
        start: Position of the shift as a fraction of the window, in
            ``[0, 1]``.

    Returns:
        ``floor(start * length)``: 0 shifts the whole window, ``length`` (for
        ``start = 1``) shifts nothing.
    """
    return math.floor(start * length)


def level_shift(xs: np.ndarray, offset: float, start: float = 0.5) -> np.ndarray:
    """Add ``offset`` to every channel from the step at ``start`` on.

    The steps before ``shift_step(W, start)`` are unchanged and every step
    from it on moves by ``offset``: an abrupt break that stays, where
    ``drift`` is a gradual ramp. ``start`` is a fraction of the window, so the
    break sits at the same relative place whatever the window length. The
    support is **pointwise**: the added value at a step depends on that step's
    position only.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``, floating point.
        offset: Change of level, in the units of the signal; negative values
            shift down.
        start: Position of the break as a fraction of the window, in
            ``[0, 1]``.

    Returns:
        The batch with the shift added, same shape and dtype.

    Raises:
        ValueError: If ``xs`` is not a floating-point ``(B, W, C)`` stack.
    """
    require_windows(xs, "level_shift")
    step = np.zeros(xs.shape[1], dtype=np.float64)
    step[shift_step(xs.shape[1], start) :] = offset
    return (xs + step[None, :, None]).astype(xs.dtype, copy=False)
