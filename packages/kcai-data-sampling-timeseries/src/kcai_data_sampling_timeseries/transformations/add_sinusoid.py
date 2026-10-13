"""Add a sinusoid of a given frequency: the algorithm math.

``add_sinusoid(xs, frequency, amplitude, phase)`` is the pure numpy map; the
transformation class that wraps it into the unary contract lives in
``kcai_data_sampling_timeseries.api.transformations.add_sinusoid``.
"""

import numpy as np

from kcai_data_sampling_timeseries.transformations._checks import require_windows


def add_sinusoid(xs: np.ndarray, frequency: float, amplitude: float, phase: float = 0.0) -> np.ndarray:
    """Add ``amplitude * sin(2π * frequency * t + phase)`` to every channel.

    ``t`` counts the steps of each window from 0, so the phase is relative to
    the window's first step: two overlapping windows get the sinusoid at
    different phases on the steps they share. The support is **pointwise**: the
    added value at a step depends on that step's position only.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``, floating point.
        frequency: Frequency of the sinusoid, in cycles per step, in
            ``(0, 0.5]``.
        amplitude: Peak amplitude, in the units of the signal, ``>= 0``.
        phase: Phase at the window's first step, in radians.

    Returns:
        The batch with the sinusoid added, same shape and dtype.

    Raises:
        ValueError: If ``xs`` is not a floating-point ``(B, W, C)`` stack.
    """
    require_windows(xs, "add_sinusoid")
    steps = np.arange(xs.shape[1])
    wave = amplitude * np.sin(2 * np.pi * frequency * steps + phase)
    return (xs + wave[None, :, None]).astype(xs.dtype, copy=False)
