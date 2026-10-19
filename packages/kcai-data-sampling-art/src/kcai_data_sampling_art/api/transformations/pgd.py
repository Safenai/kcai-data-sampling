"""The ``art_pgd`` and ``art_mim`` transformation classes.

Both run an iterative gradient attack from ART against the user's target
model through :class:`TargetEstimator`, in kcai's normalized ``[0, 1]`` units:
``max_iter`` steps of ``step_fraction * epsilon`` along the target gradient,
each projected back into the ``epsilon`` ball (L-infinity or L2). ``art_pgd``
is ART's ``ProjectedGradientDescentNumpy`` without its random start (ART
allows it on classifiers only), so it is deterministic; ``art_mim`` is ART's
``MomentumIterativeMethod``, which accumulates the previous directions.
"""

from typing import Any

from art.attacks.evasion import MomentumIterativeMethod, ProjectedGradientDescentNumpy
from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_art.api.transformations._common import to_integer
from kcai_data_sampling_art.configs import ArtMimTransformationConfig, ArtPgdTransformationConfig
from kcai_data_sampling_art.estimator import TargetEstimator


class ArtPgd(UnaryTransformation):
    """Projected gradient descent computed by ART against the user's target model.

    Same declarations as ``art_fgsm``: target role (``grad``), clips, not
    reversible, deterministic. Works on any target with ``grad``, detectors
    included.
    """

    algorithm = "art_pgd"

    #: The registered config schema this algorithm validates against (``Any``
    #: as on the base, so ``art_mim`` can register its own).
    Config: Any = ArtPgdTransformationConfig

    model_role = "target"
    model_methods = ("grad",)

    clips = True
    reversible = False

    def _attack(self, estimator: TargetEstimator, batch_size: int) -> Any:
        """Build the ART attack from the resolved parameters.

        Args:
            estimator: The target model, seen by ART.
            batch_size: The number of images attacked together.

        Returns:
            The ART attack.
        """
        epsilon = self.params["epsilon"]
        return ProjectedGradientDescentNumpy(
            estimator,
            norm=np.inf if self.params["norm"] == "inf" else 2,
            eps=epsilon,
            eps_step=self.params["step_fraction"] * epsilon,
            max_iter=self.params["max_iter"],
            num_random_init=0,
            batch_size=batch_size,
            verbose=False,
        )

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, h, w, c) uint8
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Take ``max_iter`` projected steps along the target gradient, via ART.

        Args:
            xs: Integer batch of sample arrays shaped ``(B, *sample)``.
            rngs: Unused for this deterministic operation.

        Returns:
            A batch of the same shape and dtype.
        """
        del rngs
        normalized = xs.astype(np.float32) / np.iinfo(xs.dtype).max
        attack = self._attack(TargetEstimator(self.target_model, input_shape=xs.shape[1:]), len(xs))
        # The target's loss needs no labels; ART only threads ``y`` through.
        adversarial: np.ndarray = attack.generate(normalized, y=np.zeros(len(xs)))
        return to_integer(adversarial, xs)


class ArtMim(ArtPgd):
    """Momentum iterative method computed by ART against the user's target model.

    ``art_pgd`` whose steps follow the gradient plus ``decay`` times the
    accumulated previous directions, which escapes poor local maxima.
    """

    algorithm = "art_mim"

    #: The registered config schema this algorithm validates against.
    Config = ArtMimTransformationConfig

    @override
    def _attack(self, estimator: TargetEstimator, batch_size: int) -> Any:
        epsilon = self.params["epsilon"]
        return MomentumIterativeMethod(
            estimator,
            norm=np.inf if self.params["norm"] == "inf" else 2,
            eps=epsilon,
            eps_step=self.params["step_fraction"] * epsilon,
            decay=self.params["decay"],
            max_iter=self.params["max_iter"],
            batch_size=batch_size,
            verbose=False,
        )
