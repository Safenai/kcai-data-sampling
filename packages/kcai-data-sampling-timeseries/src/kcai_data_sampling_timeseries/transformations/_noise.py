"""Shared pieces of the time-series noises: how a noise is applied, and the generator check.

Every noise draws one ``(W, C)`` array per window from that window's own
generator, then applies it in one of two modes:

- ``additive``: ``x + n``, the noise in the units of the signal;
- ``multiplicative``: ``x * (1 + n)``, the noise as a relative change, so it
  scales with the level of the signal (``n = 0.05`` is 5 %).
"""

from typing import Literal

import numpy as np

#: How a noise is applied to the signal.
NoiseMode = Literal["additive", "multiplicative"]


def require_generators(xs: np.ndarray, rngs: list[np.random.Generator], name: str) -> None:
    """Refuse a generator count that does not match the batch.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``.
        rngs: One generator per window.
        name: Algorithm name, for the error message.

    Raises:
        ValueError: If there is not exactly one generator per window.
    """
    if len(rngs) != len(xs):
        raise ValueError(f"{name}: expected one generator per window, got {len(rngs)} for {len(xs)}")


def apply_noise(xs: np.ndarray, noise: np.ndarray, mode: str) -> np.ndarray:
    """Apply a noise to the batch, additively or multiplicatively.

    Args:
        xs: Batch of windows shaped ``(B, W, C)``, floating point.
        noise: Noise of the same shape.
        mode: ``additive`` or ``multiplicative``.

    Returns:
        The noisy batch, same shape and dtype.

    Raises:
        ValueError: If ``mode`` is neither ``additive`` nor ``multiplicative``.
    """
    out: np.ndarray
    if mode == "additive":
        out = xs + noise
    elif mode == "multiplicative":
        out = xs * (1.0 + noise)
    else:
        raise ValueError(f"unknown noise mode {mode!r}: expected 'additive' or 'multiplicative'")
    return out.astype(xs.dtype, copy=False)
