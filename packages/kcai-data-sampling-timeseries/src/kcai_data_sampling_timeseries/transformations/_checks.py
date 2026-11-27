"""Input checks shared by the time-series math functions."""

import numpy as np


def require_windows(xs: np.ndarray, algorithm: str) -> None:
    """Refuse anything but a floating-point ``(B, W, C)`` window stack.

    Args:
        xs: The batch handed to the algorithm.
        algorithm: The algorithm name, for the error message.

    Raises:
        ValueError: If ``xs`` is not 3-D, or not floating point.
    """
    if xs.ndim != 3:
        raise ValueError(f"{algorithm}: expected a (B, W, C) window stack, got shape {xs.shape}")
    if not np.issubdtype(xs.dtype, np.floating):
        raise ValueError(f"{algorithm}: expected floating-point windows, got {xs.dtype}")
