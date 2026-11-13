"""Add a linear drift: the algorithm math.

``drift(xs, slope)`` is the pure numpy map; the transformation class that wraps
it into the unary contract lives in
``kcai_data_sampling_timeseries.api.transformations.drift``.
"""

import numpy as np

from kcai_data_sampling_timeseries.transformations._checks import require_windows


def drift(xs: np.ndarray, slope: float) -> np.ndarray:
    """Add ``slope * t`` to every channel.

    ``t`` counts the steps of each window from 0, so the drift is 0 on the
    window's first step and ``slope * (W - 1)`` on its last: two overlapping
    windows get different offsets on the steps they share. The support is
    **pointwise**: the added value at a step depends on that step's position
    only.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``, floating point.
        slope: Change added per step, in the units of the signal; negative
            values drift down.

    Returns:
        The batch with the drift added, same shape and dtype.

    Raises:
        ValueError: If ``xs`` is not a floating-point ``(B, W, C)`` stack.
    """
    require_windows(xs, "drift")
    ramp = np.arange(xs.shape[1], dtype=np.float64) * slope
    return (xs + ramp[None, :, None]).astype(xs.dtype, copy=False)
