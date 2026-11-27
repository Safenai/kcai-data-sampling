"""Shift the level of the whole window: the algorithm math.

``level_shift(xs, offset)`` is the pure numpy map; the transformation class
that wraps it into the unary contract lives in
``kcai_data_sampling_timeseries.api.transformations.level_shift``.
"""

import numpy as np

from kcai_data_sampling_timeseries.transformations._checks import require_windows


def level_shift(xs: np.ndarray, offset: float) -> np.ndarray:
    """Add ``offset`` to every step and every channel of each window.

    The whole window moves by ``offset``: a constant bias, such as a sensor
    or calibration offset. Like every time-series transformation for now, it
    acts on the whole sample. A break at a given step (zero before, ``offset``
    after) will come from a ``support`` parameter common to all
    transformations (an onset and a length on the series clock), not from a
    position parameter of this one. The support is **pointwise**: the added
    value at a step does not depend on any other step.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``, floating point.
        offset: Change of level, in the units of the signal; negative values
            shift down.

    Returns:
        The batch with the offset added, same shape and dtype.

    Raises:
        ValueError: If ``xs`` is not a floating-point ``(B, W, C)`` stack.
    """
    require_windows(xs, "level_shift")
    return (xs + offset).astype(xs.dtype, copy=False)
