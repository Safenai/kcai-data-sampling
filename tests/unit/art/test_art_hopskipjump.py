"""The ``ArtHopSkipJump`` transformation: ART's decision-based attack through ``TargetClassifier``.

The stub target is the two-class "brightness" classifier of the Square tests,
of which only the top class is used. HopSkipJump has no budget, so the checks
cover the declarations, the flip, the untouched alpha plane, the image left
unchanged when no starting point exists, seed reproducibility independent of
the batch, the restored global random state, and the batched queries.
"""

import numpy as np
import pytest

pytest.importorskip("kcai_data_sampling_art")

pytestmark = pytest.mark.art

from kcai_data_sampling_art.api.transformations.hopskipjump import ArtHopSkipJump  # noqa: E402

IDS = ["a", "b", "c"]


class BrightnessClassifier:
    """Two classes, ``[bright, dark]``, scored by the mean RGB value; ``predict`` only."""

    name = "brightness"

    def __init__(self) -> None:
        self.largest_batch = 0

    def predict(self, xs: np.ndarray) -> np.ndarray:
        self.largest_batch = max(self.largest_batch, len(xs))
        bright = xs[..., :3].mean(axis=(1, 2, 3))
        return np.stack([bright, 1 - bright], axis=1)


@pytest.fixture
def images() -> np.ndarray:
    """Three RGBA uint8 images, the last one just above the bright/dark boundary."""
    xs = np.random.default_rng(0).integers(0, 256, size=(3, 16, 24, 4), dtype=np.uint8)
    xs[2, ..., :3] = 128  # mean 128/255 ≈ 0.502: "bright", half a grey level above "dark"
    return xs


def hsj(target: BrightnessClassifier | None = None, seed: int = 0, **params) -> ArtHopSkipJump:
    return ArtHopSkipJump({"target_model": target or BrightnessClassifier(), "seed": seed, **params})


def test_declarations_black_box_target_stochastic() -> None:
    """Target role on ``predict`` only, stochastic (a seed is recorded), adversarial."""
    attack = hsj()
    assert attack.algorithm == "art_hopskipjump"
    assert attack.model_methods == ("predict",)
    assert attack._family() == "adversarial"
    assert attack.stochastic is True
    assert attack.seed == 0
    assert attack.params == {"norm": 2, "max_iter": 10, "max_eval": 100, "init_eval": 10, "init_size": 100}


def test_flips_every_image_and_keeps_alpha(images: np.ndarray) -> None:
    """Every image ends up in the other class; alpha untouched; the near-boundary one barely moves."""
    attack = hsj()
    out = attack.apply(images, attack.rngs(IDS))
    assert out.shape == images.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[..., 3], images[..., 3])
    classifier = BrightnessClassifier()
    before = classifier.predict(images / 255).argmax(axis=1)
    after = classifier.predict(out / 255).argmax(axis=1)
    assert (before != after).all()
    assert np.abs(out[2].astype(int) - images[2]).max() <= 8


def test_no_starting_point_leaves_the_image_unchanged(images: np.ndarray) -> None:
    """A classifier that never changes its answer gives no starting point: the batch comes back as is."""

    class Constant(BrightnessClassifier):
        def predict(self, xs: np.ndarray) -> np.ndarray:
            return np.tile([0.9, 0.1], (len(xs), 1))

    attack = hsj(Constant(), init_size=5)
    np.testing.assert_array_equal(attack.apply(images, attack.rngs(IDS)), images)


def test_reproducible_and_independent_of_the_batch(images: np.ndarray) -> None:
    """Same seed and ids give the same images, whether attacked together or one by one."""
    together = hsj().apply(images, hsj().rngs(IDS))
    alone = [hsj().apply(images[i : i + 1], hsj().rngs([IDS[i]]))[0] for i in range(3)]
    np.testing.assert_array_equal(together, np.stack(alone))


def test_restores_the_global_random_state(images: np.ndarray) -> None:
    """The attack seeds numpy's global state per image, then puts the caller's back."""
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    attack = hsj()
    attack.apply(images, attack.rngs(IDS))
    assert np.random.rand() == expected


def test_queries_are_batched(images: np.ndarray) -> None:
    """The target never receives more than 8 images per call, however many noisy copies ART builds."""
    target = BrightnessClassifier()
    attack = hsj(target, max_eval=200, init_eval=100)
    attack.apply(images, attack.rngs(IDS))
    assert 0 < target.largest_batch <= 8


def test_refuses_init_eval_above_max_eval() -> None:
    """ART would refuse it mid-run; the config refuses it at load."""
    with pytest.raises(ValueError, match="init_eval"):
        hsj(max_eval=10, init_eval=20)
