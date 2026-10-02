"""Pure algorithm math for the adversarial transformations.

All transformations share the image-batch convention ``(B, H, W, 4) uint8``.
The module-level functions here are the pure algorithm math; the
transformation class that wraps them into the unary contract lives in
``kcai_data_sampling_fgsm.api.transformations``.
"""

from kcai_data_sampling_fgsm.transformations.fgsm import fgsm_step

__all__ = ["fgsm_step"]
