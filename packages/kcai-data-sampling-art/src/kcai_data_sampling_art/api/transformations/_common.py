"""Helpers shared by the ART-backed transformations.

ART attacks draw from numpy's global random state, and work on float batches
in ``[0, 1]``; kcai hands integer batches and one generator per image.
"""

from collections.abc import Callable

import numpy as np


def per_image_seeded(
    normalized: np.ndarray,
    rngs: list[np.random.Generator],
    attack_one: Callable[[np.ndarray, np.random.Generator], np.ndarray],
) -> np.ndarray:
    """Attack each image alone, with numpy's global state seeded from its own generator.

    A result then does not depend on the batch around it, and the caller's
    global random state is restored afterwards.

    Args:
        normalized: Float batch ``(B, H, W, C)`` in ``[0, 1]``.
        rngs: One generator per image.
        attack_one: Attacks a batch of one image; also receives that image's
            generator, for any draw the attack lets the caller make.

    Returns:
        The attacked float batch, same shape.
    """
    adversarial = np.empty_like(normalized)
    state = np.random.get_state()
    try:
        for i, rng in enumerate(rngs):
            np.random.seed(int(rng.integers(2**32)))
            adversarial[i] = attack_one(normalized[i : i + 1], rng)[0]
    finally:
        np.random.set_state(state)
    return adversarial


def to_integer(adversarial: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Back to the batch's integer dtype; an alpha plane is restored from ``xs``.

    Args:
        adversarial: Float batch in ``[0, 1]``.
        xs: The original integer batch.

    Returns:
        A batch of the same shape and dtype as ``xs``.
    """
    out = np.rint(adversarial * np.iinfo(xs.dtype).max).astype(xs.dtype)
    if xs.shape[-1] == 4:
        out[..., 3] = xs[..., 3]  # the attack is on the image, not its transparency
    return out
