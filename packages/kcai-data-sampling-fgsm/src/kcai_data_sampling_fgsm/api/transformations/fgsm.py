"""The ``fgsm`` transformation class.

Wraps the normalized sign-direction step
(:func:`kcai_data_sampling_fgsm.transformations.fgsm.fgsm_step`) into the
unary transformation contract. The target model — the user's — is queried on
the normalized float batch and its gradient is funneled through
``check_output`` (shape and finiteness) before the step.
"""

from kcai_data_sampling_core.api.roles import check_output
from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_fgsm.configs import FgsmTransformationConfig
from kcai_data_sampling_fgsm.transformations.fgsm import fgsm_step


class Fgsm(UnaryTransformation):
    """One budgeted step along the sign of the target model's gradient, in L-infinity.

    The ``epsilon``-signed step may push pixels past the selection's value
    range, so the algorithm declares ``clips`` and the base clips the output
    back to it. The step is a lossy perturbation (not ``reversible``) and
    deterministic, so the row records no seed. The target model is the user's
    ``TargetModel``: only its ``grad`` is used.
    """

    algorithm = "fgsm"

    #: The registered config schema this algorithm validates against.
    Config = FgsmTransformationConfig

    model_role = "target"
    model_methods = ("grad",)

    clips = True
    reversible = False

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, h, w, c) uint8
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Perturb the batch by ``epsilon`` along the sign of the target gradient.

        Args:
            xs: Batch of sample arrays shaped ``(B, *sample)``.
            rngs: Unused for this deterministic operation.

        Returns:
            A batch of the same shape and dtype, clipped within the value range.

        Raises:
            ValueError: If the target model's ``grad`` output breaks the
                numeric contract (wrong shape or non-finite).
        """
        del rngs
        normalized = xs.astype(np.float64) / np.iinfo(xs.dtype).max
        grad = check_output(
            self.algorithm,
            self.target_model,
            "grad",
            normalized,
            self.target_model.grad(normalized),
        )
        return fgsm_step(xs, grad, self.params["epsilon"])
