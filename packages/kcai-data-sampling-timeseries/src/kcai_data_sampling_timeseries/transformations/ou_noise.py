"""Ornstein-Uhlenbeck noise: the algorithm math.

``ou_noise(xs, sigma, theta, rngs, mode)`` is the pure numpy map; the
transformation class that wraps it into the unary contract lives in
``kcai_data_sampling_timeseries.api.transformations.ou_noise``.
"""

import numpy as np

from kcai_data_sampling_timeseries.transformations._checks import require_windows
from kcai_data_sampling_timeseries.transformations._noise import apply_noise, require_generators


def ou_series(rng: np.random.Generator, length: int, channels: int, sigma: float, theta: float) -> np.ndarray:
    """A stationary Ornstein-Uhlenbeck path, sampled exactly at every step.

    ``n[0] ~ N(0, sigma²)`` and ``n[t] = a n[t-1] + sigma sqrt(1 - a²) z[t]``
    with ``a = exp(-theta)``: the path reverts to 0 at rate ``theta`` per step
    and keeps a stationary standard deviation ``sigma``.

    This is the noise model for **one** window, drawn from that window's own
    generator. ``ou_noise`` is the batch side: it checks the batch, calls this
    function once per window with that window's generator (so a window's noise
    never depends on the batch it lands in), and applies the result additively
    or multiplicatively. The two are kept apart so that the model (the
    step-by-step recursion) can be read and checked on its own, without the
    batch code around it. ``length`` is any number of steps: a window's, or a
    whole series'.

    Args:
        rng: The window's generator.
        length: Number of steps ``W``.
        channels: Number of channels ``C``.
        sigma: Stationary standard deviation.
        theta: Mean-reversion rate per step, ``> 0``; ``1 / theta`` is the
            correlation time in steps.

    Returns:
        ``(W, C)`` array.
    """
    a = np.exp(-theta)
    shocks = rng.standard_normal((length, channels))
    path = np.empty((length, channels))
    path[0] = sigma * shocks[0]
    step = sigma * np.sqrt(1.0 - a * a)
    for t in range(1, length):
        path[t] = a * path[t - 1] + step * shocks[t]
    return path


def ou_noise(
    xs: np.ndarray, sigma: float, theta: float, rngs: list[np.random.Generator], mode: str = "additive"
) -> np.ndarray:
    """Add (or multiply by) Ornstein-Uhlenbeck noise.

    A mean-reverting random walk: the noise wanders away from 0 and is pulled
    back at rate ``theta``, so successive steps are correlated over about
    ``1 / theta`` steps, like a slowly fluctuating sensor offset. Each window
    and channel draws its own stationary path. The support is **causal**: the
    noise at a step depends on the draws up to that step.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``, floating point.
        sigma: Stationary standard deviation, ``>= 0``; signal units in
            ``additive`` mode, a relative change in ``multiplicative`` mode.
        theta: Mean-reversion rate per step, ``> 0``.
        rngs: One generator per window, in batch order.
        mode: ``additive`` or ``multiplicative``.

    Returns:
        The noisy batch, same shape and dtype.

    Raises:
        ValueError: If ``xs`` is not a floating-point ``(B, W, C)`` stack, if
            there is not exactly one generator per window, or if ``mode`` is
            unknown.
    """
    require_windows(xs, "ou_noise")
    require_generators(xs, rngs, "ou_noise")
    _, w, c = xs.shape
    noise = np.stack([ou_series(rng, w, c, sigma, theta) for rng in rngs]) if len(xs) else np.zeros_like(xs)
    return apply_noise(xs, noise, mode)
