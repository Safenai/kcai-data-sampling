"""Impulse noise, isolated spikes ("salt and pepper"): the algorithm math.

``impulse_noise(xs, rate, amplitude, rngs, mode)`` is the pure numpy map; the
transformation class that wraps it into the unary contract lives in
``kcai_data_sampling_timeseries.api.transformations.impulse_noise``.
"""

import numpy as np

from kcai_data_sampling_timeseries.transformations._checks import require_windows
from kcai_data_sampling_timeseries.transformations._noise import apply_noise, require_generators


def draw_impulse_noise(
    rng: np.random.Generator, length: int, channels: int, rate: float, amplitude: float
) -> np.ndarray:
    """Spikes of size ``amplitude`` and random sign, at random steps.

    Each step of each channel is hit independently with probability ``rate``
    (Bernoulli arrivals, the discrete counterpart of a Poisson process), and a
    hit is ``+amplitude`` or ``-amplitude`` with equal probability.

    This is the noise model for **one** window, drawn from that window's own
    generator. ``impulse_noise`` is the batch side: it checks the batch, calls
    this function once per window with that window's generator (so a window's
    noise never depends on the batch it lands in), and applies the result
    additively or multiplicatively. The two are kept apart so that the model
    (the arrivals and their signs) can be read and checked on its own, without
    the batch code around it. ``length`` is any number of steps: a window's, or
    a whole series'.

    Args:
        rng: The window's generator.
        length: Number of steps ``W``.
        channels: Number of channels ``C``.
        rate: Probability of a spike at each step, in ``[0, 1]``.
        amplitude: Size of a spike, ``>= 0``.

    Returns:
        ``(W, C)`` array, zero except at the spikes.
    """
    hits = rng.random((length, channels)) < rate
    signs = rng.choice((-1.0, 1.0), size=(length, channels))
    return np.where(hits, amplitude * signs, 0.0)


def impulse_noise(
    xs: np.ndarray, rate: float, amplitude: float, rngs: list[np.random.Generator], mode: str = "additive"
) -> np.ndarray:
    """Add (or multiply by) isolated spikes.

    Most steps are untouched; a few, about ``rate * W`` per window and
    channel, jump by ``±amplitude``, like glitches of an acquisition chain.
    Each window and channel draws its own spikes. The support is
    **pointwise**: the noise at a step depends on that step's draw only.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``, floating point.
        rate: Probability of a spike at each step, in ``[0, 1]``.
        amplitude: Size of a spike, ``>= 0``; signal units in ``additive``
            mode, a relative change in ``multiplicative`` mode.
        rngs: One generator per window, in batch order.
        mode: ``additive`` or ``multiplicative``.

    Returns:
        The noisy batch, same shape and dtype.

    Raises:
        ValueError: If ``xs`` is not a floating-point ``(B, W, C)`` stack, if
            there is not exactly one generator per window, or if ``mode`` is
            unknown.
    """
    require_windows(xs, "impulse_noise")
    require_generators(xs, rngs, "impulse_noise")
    _, w, c = xs.shape
    noise = np.stack([draw_impulse_noise(rng, w, c, rate, amplitude) for rng in rngs]) if len(xs) else np.zeros_like(xs)
    return apply_noise(xs, noise, mode)
