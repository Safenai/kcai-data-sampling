"""Coloured noise, power spectrum in 1/f^exponent: the algorithm math.

``colored_noise(xs, sigma, exponent, rngs, mode)`` is the pure numpy map; the
transformation class that wraps it into the unary contract lives in
``kcai_data_sampling_timeseries.api.transformations.colored_noise``.
"""

import numpy as np

from kcai_data_sampling_timeseries.transformations._checks import require_windows
from kcai_data_sampling_timeseries.transformations._noise import apply_noise, require_generators


def draw_colored_noise(rng: np.random.Generator, length: int, channels: int, exponent: float) -> np.ndarray:
    """Zero-mean, unit-variance noise whose power spectrum falls as 1/f^exponent.

    White Gaussian noise is shaped in the frequency domain: each frequency's
    amplitude is scaled by ``f^(-exponent / 2)``, the constant term is
    removed, and the result is rescaled to unit variance per channel.

    This is the noise model for **one** window, drawn from that window's own
    generator. ``colored_noise`` is the batch side: it checks the batch, calls
    this function once per window with that window's generator (so a window's
    noise never depends on the batch it lands in), and applies the result
    additively or multiplicatively. The two are kept apart so that the model
    (the spectral shaping) can be read and checked on its own, without the
    batch code around it. ``length`` is any number of steps: a window's, or a
    whole series'.

    Args:
        rng: The window's generator.
        length: Number of steps ``W``.
        channels: Number of channels ``C``.
        exponent: Spectral exponent: 0 white, 1 pink, 2 brown (red).

    Returns:
        ``(W, C)`` array.
    """
    white = rng.standard_normal((length, channels))
    freqs = np.fft.rfftfreq(length)
    scale = np.zeros_like(freqs)
    scale[1:] = freqs[1:] ** (-exponent / 2)
    shaped = np.fft.irfft(np.fft.rfft(white, axis=0) * scale[:, None], n=length, axis=0)
    std = shaped.std(axis=0, keepdims=True)
    return np.divide(shaped, std, out=np.zeros_like(shaped), where=std > 0)


def colored_noise(
    xs: np.ndarray, sigma: float, exponent: float, rngs: list[np.random.Generator], mode: str = "additive"
) -> np.ndarray:
    """Add (or multiply by) coloured noise of standard deviation ``sigma``.

    Unlike ``jitter``, the noise is correlated in time: with ``exponent`` 1
    (pink) or 2 (brown), slow wanderings dominate, like a drifting sensor.
    Each window and channel draws its own noise, rescaled to exactly ``sigma``
    over the window. The support is **global**: the noise at one step depends
    on the draw of the whole window.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``, floating point.
        sigma: Standard deviation of the noise, ``>= 0``; signal units in
            ``additive`` mode, a relative change in ``multiplicative`` mode.
        exponent: Spectral exponent in ``[0, 2]``: 0 white, 1 pink, 2 brown.
        rngs: One generator per window, in batch order.
        mode: ``additive`` or ``multiplicative``.

    Returns:
        The noisy batch, same shape and dtype.

    Raises:
        ValueError: If ``xs`` is not a floating-point ``(B, W, C)`` stack, if
            there is not exactly one generator per window, or if ``mode`` is
            unknown.
    """
    require_windows(xs, "colored_noise")
    require_generators(xs, rngs, "colored_noise")
    _, w, c = xs.shape
    noise = (
        np.stack([sigma * draw_colored_noise(rng, w, c, exponent) for rng in rngs]) if len(xs) else np.zeros_like(xs)
    )
    return apply_noise(xs, noise, mode)
