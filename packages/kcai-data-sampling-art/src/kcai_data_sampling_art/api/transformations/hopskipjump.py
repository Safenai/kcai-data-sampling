"""The ``art_hopskipjump`` transformation class.

Runs ART's ``HopSkipJump`` (decision-based, black-box) against the user's
target model through :class:`TargetClassifier`: only the model's top class is
used. It starts from a noise image the model already classifies differently,
then alternates a gradient estimate (from the top class of noisy copies), a
step, and a binary search back toward the original image's decision boundary.
There is no budget: the result is the closest misclassified image it finds.
"""

from art.attacks.evasion import HopSkipJump
from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_art.api.transformations._common import per_image_seeded, to_integer
from kcai_data_sampling_art.configs import ArtHopSkipJumpTransformationConfig
from kcai_data_sampling_art.estimator import TargetClassifier

#: Most noisy copies sent to the target per ``predict`` call. A memory knob
#: only (the result does not depend on it): each copy is a full-size image.
_QUERY_BATCH = 8


class ArtHopSkipJump(UnaryTransformation):
    """HopSkipJump computed by ART against the user's target model.

    ART draws its starting noise image from an unseeded generator, so this
    class draws it instead, from the image's own generator: the result is
    reproducible from the seed. The alpha plane is masked out of the attack.
    """

    algorithm = "art_hopskipjump"

    #: The registered config schema this algorithm validates against.
    Config = ArtHopSkipJumpTransformationConfig

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
        """Attack each image of the batch alone; an image with no starting point is returned as is.

        Args:
            xs: Integer batch of sample arrays shaped ``(B, H, W, C)``.
            rngs: One generator per image.

        Returns:
            A batch of the same shape and dtype.
        """
        assert rngs is not None
        normalized = xs.astype(np.float32) / np.iinfo(xs.dtype).max
        classifier = TargetClassifier(self.target_model, input_shape=xs.shape[1:])
        attack = HopSkipJump(
            classifier,
            norm=np.inf if self.params["norm"] == "inf" else 2,
            max_iter=self.params["max_iter"],
            max_eval=self.params["max_eval"],
            init_eval=self.params["init_eval"],
            init_size=self.params["init_size"],
            batch_size=_QUERY_BATCH,
            verbose=False,
        )
        mask = np.ones(xs.shape[1:], dtype=np.float32)
        if xs.shape[-1] == 4:
            mask[..., 3] = 0  # the attack is on the image, not its transparency

        def attack_one(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            start = self._starting_point(classifier, x, mask, rng)
            if start is None:
                return x
            adversarial: np.ndarray = attack.generate(x, x_adv_init=start, mask=mask)
            return adversarial

        return to_integer(per_image_seeded(normalized, rngs, attack_one), xs)

    def _starting_point(
        self, classifier: TargetClassifier, x: np.ndarray, mask: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray | None:
        """Draw noise images until one changes the top class, then move it back toward ``x``.

        ART's own first step, with a seeded generator: a binary search on the
        blend ``(1 - t) * x + t * noise`` keeps the smallest ``t`` that still
        changes the class.

        Args:
            classifier: The target model, seen by ART.
            x: Batch of one normalized image.
            mask: Where the image may change (1) or not (0).
            rng: The image's generator.

        Returns:
            A batch of one misclassified image, or ``None`` if no noise image
            of ``init_size`` changes the class.
        """
        label = classifier.predict(x).argmax(axis=1)[0]
        for _ in range(self.params["init_size"]):
            noise = x + mask * (rng.uniform(0, 1, size=x.shape).astype(np.float32) - x)
            if classifier.predict(noise).argmax(axis=1)[0] != label:
                break
        else:
            return None
        low, high = 0.0, 1.0  # class unchanged at low, changed at high
        while high - low > 0.001:
            middle = (low + high) / 2
            blend = (1 - middle) * x + middle * noise
            if classifier.predict(blend).argmax(axis=1)[0] != label:
                high = middle
            else:
                low = middle
        return np.asarray((1 - high) * x + high * noise, dtype=np.float32)
