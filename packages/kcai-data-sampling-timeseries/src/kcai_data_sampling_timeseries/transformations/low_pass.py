"""Remove the frequencies above a cutoff: the algorithm math.

``low_pass(xs, cutoff)`` is the pure numpy map; the transformation class that
wraps it into the unary contract lives in
``kcai_data_sampling_timeseries.api.transformations.low_pass``.
"""

import numpy as np

from kcai_data_sampling_timeseries.transformations._checks import require_windows


def low_pass(xs: np.ndarray, cutoff: float) -> np.ndarray:
    """Zero every frequency above ``cutoff``, per window and per channel.

    The filter is an ideal (brick-wall) one, computed with a real FFT along the
    time axis of each window: the components at or below ``cutoff`` are kept
    unchanged, the others are set to zero. Its support is **global**: the output
    at one step depends on the whole window, so a window cut elsewhere gives a
    different result.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``, floating point.
        cutoff: Highest frequency kept, in cycles per step, in ``(0, 0.5]``;
            ``0.5`` (Nyquist) keeps everything.

    Returns:
        The filtered batch, same shape and dtype.

    Raises:
        ValueError: If ``xs`` is not a floating-point ``(B, W, C)`` stack.
    """
    require_windows(xs, "low_pass")
    width = xs.shape[1]
    spectrum = np.fft.rfft(xs, axis=1)
    spectrum[:, np.fft.rfftfreq(width) > cutoff, :] = 0
    return np.fft.irfft(spectrum, n=width, axis=1).astype(xs.dtype, copy=False)
