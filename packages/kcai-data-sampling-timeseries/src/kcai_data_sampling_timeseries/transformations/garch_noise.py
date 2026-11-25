"""GARCH(1,1) noise, conditional volatility: the algorithm math.

``garch_noise(xs, sigma, alpha, beta, rngs, mode)`` is the pure numpy map; the
transformation class that wraps it into the unary contract lives in
``kcai_data_sampling_timeseries.api.transformations.garch_noise``.
"""

import numpy as np

from kcai_data_sampling_timeseries.transformations._checks import require_windows
from kcai_data_sampling_timeseries.transformations._noise import apply_noise, require_generators


def garch_series(
    rng: np.random.Generator, length: int, channels: int, sigma: float, alpha: float, beta: float
) -> np.ndarray:
    """A GARCH(1,1) path with unconditional standard deviation ``sigma``.

    ``e[t] = sqrt(h[t]) z[t]`` and ``h[t+1] = omega + alpha e[t]² + beta h[t]``
    with ``omega = sigma² (1 - alpha - beta)`` and ``h[0] = sigma²``: a large
    shock raises the variance of the next steps, so calm and agitated periods
    alternate (volatility clustering).

    This is the noise model for **one** window, drawn from that window's own
    generator. ``garch_noise`` is the batch side: it checks the batch, calls
    this function once per window with that window's generator (so a window's
    noise never depends on the batch it lands in), and applies the result
    additively or multiplicatively. The two are kept apart so that the model
    (the variance recursion) can be read and checked on its own, without the
    batch code around it. ``length`` is any number of steps: a window's, or a
    whole series'.

    Args:
        rng: The window's generator.
        length: Number of steps ``W``.
        channels: Number of channels ``C``.
        sigma: Unconditional standard deviation.
        alpha: Weight of the last shock, ``>= 0``.
        beta: Persistence of the variance, ``>= 0``, with ``alpha + beta < 1``.

    Returns:
        ``(W, C)`` array.
    """
    omega = sigma * sigma * (1.0 - alpha - beta)
    shocks = rng.standard_normal((length, channels))
    path = np.empty((length, channels))
    h = np.full(channels, sigma * sigma)
    for t in range(length):
        path[t] = np.sqrt(h) * shocks[t]
        h = omega + alpha * path[t] ** 2 + beta * h
    return path


def garch_noise(
    xs: np.ndarray,
    sigma: float,
    alpha: float,
    beta: float,
    rngs: list[np.random.Generator],
    mode: str = "additive",
) -> np.ndarray:
    """Add (or multiply by) GARCH(1,1) noise.

    The noise has no memory of its sign but remembers its size: bursts of
    large values cluster in time, like a sensor or a market going through
    agitated periods. Each window and channel draws its own path. The support
    is **causal**: the noise at a step depends on the draws up to that step.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``, floating point.
        sigma: Unconditional standard deviation, ``>= 0``; signal units in
            ``additive`` mode, a relative change in ``multiplicative`` mode.
        alpha: Weight of the last shock, ``>= 0``.
        beta: Persistence of the variance, ``>= 0``, with ``alpha + beta < 1``.
        rngs: One generator per window, in batch order.
        mode: ``additive`` or ``multiplicative``.

    Returns:
        The noisy batch, same shape and dtype.

    Raises:
        ValueError: If ``xs`` is not a floating-point ``(B, W, C)`` stack, if
            there is not exactly one generator per window, if ``alpha + beta``
            is not below 1, or if ``mode`` is unknown.
    """
    require_windows(xs, "garch_noise")
    require_generators(xs, rngs, "garch_noise")
    if alpha + beta >= 1:
        raise ValueError(f"garch_noise: alpha + beta must be below 1, got {alpha + beta}")
    _, w, c = xs.shape
    noise = np.stack([garch_series(rng, w, c, sigma, alpha, beta) for rng in rngs]) if len(xs) else np.zeros_like(xs)
    return apply_noise(xs, noise, mode)
