"""The ``art_square`` transformation class.

Runs ART's ``SquareAttack`` (L-infinity, black-box) against the user's target
model through :class:`TargetClassifier`: the attack only queries the model's
``predict``, so it also works on models that cannot give a gradient.
"""

import math

from art.attacks.evasion import SquareAttack
from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_art.configs import ArtSquareTransformationConfig
from kcai_data_sampling_art.estimator import TargetClassifier


class ArtSquare(UnaryTransformation):
    """Square attack computed by ART against the user's target model.

    Random square-shaped updates of size ``±epsilon`` are kept when they lower
    the model's margin between its top class and the runner-up, until the top
    class changes or the query budget is spent. Needs only ``predict``.
    """

    algorithm = "art_square"

    #: The registered config schema this algorithm validates against.
    Config = ArtSquareTransformationConfig

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
        """Attack each image of the batch, then restore its alpha plane.

        ART draws its squares from numpy's global random state: each image is
        attacked alone, with that state seeded from the image's own generator,
        so a result does not depend on the batch around it. The caller's global
        random state is restored afterwards.

        Args:
            xs: Integer batch of sample arrays shaped ``(B, H, W, C)``.
            rngs: One generator per image (the algorithm is stochastic).

        Returns:
            A batch of the same shape and dtype.

        Raises:
            ValueError: If ``p_init`` makes the first square at least as large
                as the image's shorter side (ART would fail mid-attack).
        """
        assert rngs is not None
        height, width = xs.shape[1:3]
        side = max(round(math.sqrt(self.params["p_init"] * height * width)), 1)
        if side >= min(height, width):
            raise ValueError(
                f"art_square: p_init={self.params['p_init']} gives a first square of {side} px, "
                f"not smaller than the {height}x{width} image; lower p_init"
            )
        top = np.iinfo(xs.dtype).max
        normalized = xs.astype(np.float32) / top
        attack = SquareAttack(
            TargetClassifier(self.target_model, input_shape=xs.shape[1:]),
            norm=np.inf,
            eps=self.params["epsilon"],
            max_iter=self.params["max_iter"],
            p_init=self.params["p_init"],
            verbose=False,
        )
        adversarial = np.empty_like(normalized)
        state = np.random.get_state()
        try:
            for i, rng in enumerate(rngs):
                np.random.seed(int(rng.integers(2**32)))
                adversarial[i] = attack.generate(normalized[i : i + 1])[0]
        finally:
            np.random.set_state(state)
        out = np.rint(adversarial * top).astype(xs.dtype)
        if xs.shape[-1] == 4:
            out[..., 3] = xs[..., 3]  # the attack is on the image, not its transparency
        return out
