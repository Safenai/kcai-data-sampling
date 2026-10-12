"""The ``ArtSquare`` transformation: ART's black-box Square attack through ``TargetClassifier``.

The stub target only has ``predict`` (no gradient): a two-class "brightness"
classifier whose scores are the image's mean RGB value and its complement. The
checks cover the declarations, the L-infinity budget, the untouched alpha
plane, seed reproducibility independent of the batch, the restored global
random state, and the refusals (no ``predict``, malformed scores).
"""

import numpy as np
import pytest

pytest.importorskip("kcai_data_sampling_art")

pytestmark = pytest.mark.art

from kcai_data_sampling_art.api.transformations.square import ArtSquare  # noqa: E402

EPSILON = 8 / 255


class BrightnessClassifier:
    """Two classes, ``[bright, dark]``, scored by the mean RGB value; ``predict`` only."""

    name = "brightness"

    def predict(self, xs: np.ndarray) -> np.ndarray:
        bright = xs[..., :3].mean(axis=(1, 2, 3))
        return np.stack([bright, 1 - bright], axis=1)


@pytest.fixture
def images() -> np.ndarray:
    """Three RGBA uint8 images, the last one just above the bright/dark boundary."""
    rng = np.random.default_rng(0)
    xs = rng.integers(0, 256, size=(3, 16, 24, 4), dtype=np.uint8)
    xs[2, ..., :3] = 128  # mean 128/255 ≈ 0.502: "bright", half a grey level above "dark"
    return xs


def square(seed: int = 0, **params) -> ArtSquare:
    return ArtSquare({"target_model": BrightnessClassifier(), "epsilon": EPSILON, "seed": seed, **params})


def test_declarations_black_box_target_stochastic() -> None:
    """Target role on ``predict`` only, stochastic (a seed is recorded), adversarial."""
    attack = square()
    assert attack.algorithm == "art_square"
    assert attack.model_role == "target"
    assert attack.model_methods == ("predict",)
    assert attack._family() == "adversarial"
    assert attack.stochastic is True
    assert attack.seed == 0
    assert attack.params == {"epsilon": EPSILON, "max_iter": 100, "p_init": 0.05}


def test_stays_within_budget_and_keeps_alpha(images: np.ndarray) -> None:
    """Same shape and dtype, at most 8 grey levels of change, alpha untouched."""
    attack = square()
    out = attack.apply(images, attack.rngs(["a", "b", "c"]))
    assert out.shape == images.shape
    assert out.dtype == np.uint8
    assert np.abs(out.astype(int) - images).max() <= round(EPSILON * 255)
    np.testing.assert_array_equal(out[..., 3], images[..., 3])


def test_flips_an_image_near_the_boundary(images: np.ndarray) -> None:
    """The image just above the boundary ends up classified "dark".

    Square is a random search: success depends on the draw. With 1000 queries
    it flips this image for 19 seeds out of 20; seed 0 is one of them, and the
    seeded attack is deterministic.
    """
    attack = square(seed=0, max_iter=1000)
    out = attack.apply(images, attack.rngs(["a", "b", "c"]))
    before = BrightnessClassifier().predict(images[2:] / 255).argmax(axis=1)
    after = BrightnessClassifier().predict(out[2:] / 255).argmax(axis=1)
    assert before[0] == 0
    assert after[0] == 1


def test_reproducible_and_independent_of_the_batch(images: np.ndarray) -> None:
    """Same seed and ids give the same images, whether attacked together or one by one."""
    ids = ["a", "b", "c"]
    together = square().apply(images, square().rngs(ids))
    alone = [square().apply(images[i : i + 1], square().rngs([ids[i]]))[0] for i in range(3)]
    np.testing.assert_array_equal(together, np.stack(alone))


def test_restores_the_global_random_state(images: np.ndarray) -> None:
    """The attack seeds numpy's global state per image, then puts the caller's back."""
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    attack = square()
    attack.apply(images, attack.rngs(["a", "b", "c"]))
    assert np.random.rand() == expected


def test_refuses_a_target_without_predict() -> None:
    """A grad-only target cannot serve a black-box attack: refused at construction."""

    class GradOnly:
        name = "grad_only"

        def grad(self, xs: np.ndarray) -> np.ndarray:
            return np.zeros_like(xs)

    with pytest.raises(ValueError, match="predict"):
        ArtSquare({"target_model": GradOnly(), "epsilon": EPSILON, "seed": 0})


def test_refuses_malformed_scores(images: np.ndarray) -> None:
    """``predict`` must answer ``(B, n_classes)`` scores."""

    class OneScore(BrightnessClassifier):
        def predict(self, xs: np.ndarray) -> np.ndarray:
            return xs[..., :3].mean(axis=(1, 2, 3))  # (B,) instead of (B, 2)

    attack = ArtSquare({"target_model": OneScore(), "epsilon": EPSILON, "seed": 0})
    with pytest.raises(ValueError, match="expected class scores"):
        attack.apply(images, attack.rngs(["a", "b", "c"]))


def test_refuses_a_first_square_larger_than_the_image(images: np.ndarray) -> None:
    """ART's own default ``p_init=0.8`` would make a 18 px square on 16 px high images."""
    attack = square(p_init=0.8)
    with pytest.raises(ValueError, match="lower p_init"):
        attack.apply(images, attack.rngs(["a", "b", "c"]))
