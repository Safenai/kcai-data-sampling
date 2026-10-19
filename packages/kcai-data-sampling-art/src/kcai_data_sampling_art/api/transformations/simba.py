"""The ``art_simba`` transformation class.

Runs ART's ``SimBA`` (simple black-box attack, pixel mode) against the user's
target model through :class:`TargetClassifier`: one random pixel value at a
time is moved by ``±epsilon``, and kept when it lowers the probability of the
model's current class. Each value moves at most once, so ``epsilon`` is also
the L-infinity budget. Only ``predict`` (probabilities) is queried.
"""

from art.attacks.evasion import SimBA
from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_art.api.transformations._common import per_image_seeded, to_integer
from kcai_data_sampling_art.configs import ArtSimbaTransformationConfig
from kcai_data_sampling_art.estimator import TargetClassifier


class ArtSimba(UnaryTransformation):
    """SimBA computed by ART against the user's target model.

    ART's DCT mode assumes square images, so only the pixel mode is offered.
    Each query moves one value out of ``H * W * C``: on large images, a model
    that downsizes its input needs many queries to be affected.
    """

    algorithm = "art_simba"

    #: The registered config schema this algorithm validates against.
    Config = ArtSimbaTransformationConfig

    model_role = "target"
    model_methods = ("predict",)

    stochastic = True
    clips = True
    reversible = False

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, h, w, c) uint8
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Attack each image of the batch alone, then restore its alpha plane.

        Args:
            xs: Integer batch of sample arrays shaped ``(B, H, W, C)``.
            rngs: One generator per image (the pixel order is random).

        Returns:
            A batch of the same shape and dtype.
        """
        assert rngs is not None
        normalized = xs.astype(np.float32) / np.iinfo(xs.dtype).max
        attack = SimBA(
            TargetClassifier(self.target_model, input_shape=xs.shape[1:]),
            attack="px",
            max_iter=self.params["max_iter"],
            epsilon=self.params["epsilon"],
            order="random",
            verbose=False,
        )
        adversarial = per_image_seeded(normalized, rngs, lambda x, _rng: attack.generate(x))
        return to_integer(adversarial, xs)
