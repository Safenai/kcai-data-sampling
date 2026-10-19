"""The ``ArtSimba`` transformation: ART's black-box SimBA (pixel mode) through ``TargetClassifier``.

The stub target is the two-class "brightness" classifier of the Square tests:
``predict`` only, probabilities. The checks cover the declarations, the
L-infinity budget, the untouched alpha plane, the flip of an image near the
boundary, seed reproducibility independent of the batch, and the refusal of
scores that are not probabilities.
"""

import numpy as np
import pytest

pytest.importorskip("kcai_data_sampling_art")

pytestmark = pytest.mark.art

from kcai_data_sampling_art.api.transformations.simba import ArtSimba  # noqa: E402

EPSILON = 8 / 255
IDS = ["a", "b", "c"]


class BrightnessClassifier:
    """Two classes, ``[bright, dark]``, scored by the mean RGB value; ``predict`` only."""

    name = "brightness"

    def predict(self, xs: np.ndarray) -> np.ndarray:
        bright = xs[..., :3].mean(axis=(1, 2, 3))
        return np.stack([bright, 1 - bright], axis=1)


@pytest.fixture
def images() -> np.ndarray:
    """Three RGBA uint8 images, the last one just above the bright/dark boundary."""
    xs = np.random.default_rng(0).integers(0, 256, size=(3, 16, 24, 4), dtype=np.uint8)
    xs[2, ..., :3] = 128  # mean 128/255 ≈ 0.502: "bright", half a grey level above "dark"
    return xs


def simba(seed: int = 0, **params) -> ArtSimba:
    return ArtSimba({"target_model": BrightnessClassifier(), "epsilon": EPSILON, "seed": seed, **params})


def test_declarations_black_box_target_stochastic() -> None:
    """Target role on ``predict`` only, stochastic (a seed is recorded), adversarial."""
    attack = simba()
    assert attack.algorithm == "art_simba"
    assert attack.model_methods == ("predict",)
    assert attack._family() == "adversarial"
    assert attack.stochastic is True
    assert attack.seed == 0
    assert attack.params == {"epsilon": EPSILON, "max_iter": 1000}


def test_stays_within_budget_keeps_alpha_and_flips(images: np.ndarray) -> None:
    """At most 8 grey levels of change, alpha untouched, the image near the boundary turns "dark"."""
    attack = simba()
    out = attack.apply(images, attack.rngs(IDS))
    assert out.shape == images.shape
    assert out.dtype == np.uint8
    assert np.abs(out.astype(int) - images).max() <= round(EPSILON * 255)
    np.testing.assert_array_equal(out[..., 3], images[..., 3])
    classifier = BrightnessClassifier()
    assert classifier.predict(images[2:] / 255).argmax(axis=1)[0] == 0
    assert classifier.predict(out[2:] / 255).argmax(axis=1)[0] == 1


def test_reproducible_and_independent_of_the_batch(images: np.ndarray) -> None:
    """Same seed and ids give the same images, whether attacked together or one by one."""
    together = simba().apply(images, simba().rngs(IDS))
    alone = [simba().apply(images[i : i + 1], simba().rngs([IDS[i]]))[0] for i in range(3)]
    np.testing.assert_array_equal(together, np.stack(alone))


def test_refuses_scores_that_are_not_probabilities(images: np.ndarray) -> None:
    """SimBA compares probabilities: logits are refused by ART."""

    class Logits(BrightnessClassifier):
        def predict(self, xs: np.ndarray) -> np.ndarray:
            return 10 * super().predict(xs)

    attack = ArtSimba({"target_model": Logits(), "epsilon": EPSILON, "seed": 0})
    with pytest.raises(ValueError, match="probabilities"):
        attack.apply(images, attack.rngs(IDS))
