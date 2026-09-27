"""The FGSM step, in normalized units: the math.

``fgsm_step(xs, grad, epsilon)`` is the pure numpy map — the transformation
class that wraps it into the unary contract lives in
``kcai_data_sampling_fgsm.api.transformations.fgsm``. The step runs in
normalized ``[0, 1]`` pixel units (the batch divided by its dtype maximum), and
the model's gradient is in the same units: only its sign participates.
"""

import numpy as np


def fgsm_step(xs: np.ndarray, grad: np.ndarray, epsilon: float) -> np.ndarray:
    """Apply one budgeted sign-direction step in normalized ``[0, 1]`` units.

    Normalizes the integer batch by its dtype maximum, adds
    ``epsilon * sign(grad)``, clips back to ``[0, 1]``, and restores the
    batch's dtype (scale, round, cast). ``grad`` is the target model's gradient
    in the same normalized units — the step is ``epsilon * sign(grad)``, so
    only the sign matters.

    Args:
        xs: Integer batch of sample arrays ``(B, *sample)``.
        grad: Signed gradient in normalized units, same shape as ``xs``.
        epsilon: Adversarial budget in normalized ``[0, 1]`` pixel units,
            strictly positive.

    Returns:
        The perturbed batch, same shape and dtype as ``xs``.

    Raises:
        ValueError: If ``xs`` is not an integer array (a normalization-by-max
            has no meaning otherwise).
    """
    if not np.issubdtype(xs.dtype, np.integer):
        raise ValueError(f"fgsm_step: integer batches only, got dtype {xs.dtype}")
    info = np.iinfo(xs.dtype)
    normalized = xs.astype(np.float64) / info.max
    stepped = np.clip(normalized + epsilon * np.sign(grad), 0.0, 1.0)
    return np.rint(stepped * info.max).astype(xs.dtype)