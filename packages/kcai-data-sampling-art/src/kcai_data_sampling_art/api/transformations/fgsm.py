"""The ``art_fgsm`` transformation class.

Runs ART's ``FastGradientMethod`` (L-infinity, one step, no random start)
against the user's target model through :class:`TargetEstimator`, in kcai's
normalized ``[0, 1]`` units, then restores the batch's integer dtype.
"""

from art.attacks.evasion import FastGradientMethod
from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_art.configs import ArtFgsmTransformationConfig
from kcai_data_sampling_art.estimator import TargetEstimator


class ArtFgsm(UnaryTransformation):
    """FGSM computed by ART against the user's target model.

    Same declarations as the local ``fgsm``: target role (``grad``), clips,
    not reversible, deterministic.
    """

    algorithm = "art_fgsm"

    #: The registered config schema this algorithm validates against.
    Config = ArtFgsmTransformationConfig

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
        """Perturb the batch by ``epsilon`` along the sign of the target gradient, via ART.

        Args:
            xs: Integer batch of sample arrays shaped ``(B, *sample)``.
            rngs: Unused for this deterministic operation.

        Returns:
            A batch of the same shape and dtype.
        """
        del rngs
        top = np.iinfo(xs.dtype).max
        normalized = xs.astype(np.float32) / top
        attack = FastGradientMethod(
            TargetEstimator(self.target_model, input_shape=xs.shape[1:]),
            norm=np.inf,
            eps=self.params["epsilon"],
            batch_size=len(xs),
        )
        # The target's loss needs no labels; ART only threads ``y`` through.
        adversarial: np.ndarray = attack.generate(normalized, y=np.zeros(len(xs)))
        return np.rint(adversarial * top).astype(xs.dtype)
