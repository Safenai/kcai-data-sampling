"""Add Gaussian noise to every step: the algorithm math.

``jitter(xs, sigma, rngs)`` is the pure numpy map; the transformation class
that wraps it into the unary contract lives in
``kcai_data_sampling_timeseries.api.transformations.jitter``.
"""

import numpy as np

from kcai_data_sampling_timeseries.transformations._checks import require_windows


def jitter(xs: np.ndarray, sigma: float, rngs: list[np.random.Generator]) -> np.ndarray:
    """Add independent ``N(0, sigma²)`` noise to every step and channel.

    Each window draws its noise from its own generator, so its output never
    depends on the other windows of the batch. The support is **pointwise**:
    the noise at a step is drawn for that step alone.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``, floating point.
        sigma: Standard deviation of the noise, in the units of the signal,
            ``>= 0``.
        rngs: One generator per window, in batch order.

    Returns:
        The batch with the noise added, same shape and dtype.

    Raises:
        ValueError: If ``xs`` is not a floating-point ``(B, W, C)`` stack, or
            if there is not exactly one generator per window.
    """
    require_windows(xs, "jitter")
    if len(rngs) != len(xs):
        raise ValueError(f"jitter: expected one generator per window, got {len(rngs)} for {len(xs)}")
    noise = np.stack([rng.normal(0.0, sigma, size=xs.shape[1:]) for rng in rngs]) if len(xs) else 0.0
    return (xs + noise).astype(xs.dtype, copy=False)
