"""The ``ArtPgd`` and ``ArtMim`` transformations: ART's iterative gradient attacks through ``TargetEstimator``.

The reference is the local ``fgsm_step``: one step of the full budget is FGSM,
for both attacks. Beyond that, the checks cover the declarations, the
L-infinity and L2 budgets, the untouched alpha plane, and the independence
from the batch.
"""

import numpy as np
import pytest
from tests.fixtures.registries import StubTarget

pytest.importorskip("kcai_data_sampling_art")

pytestmark = pytest.mark.art

from kcai_data_sampling_art.api.transformations.pgd import ArtMim, ArtPgd  # noqa: E402

EPSILON = 8 / 255


@pytest.fixture
def images() -> np.ndarray:
    """Three random RGBA uint8 images."""
    return np.random.default_rng(0).integers(0, 256, size=(3, 16, 24, 4), dtype=np.uint8)


@pytest.mark.parametrize(("cls", "algorithm"), [(ArtPgd, "art_pgd"), (ArtMim, "art_mim")])
def test_declarations_gradient_target_deterministic(cls: type[ArtPgd], algorithm: str) -> None:
    """Target role on ``grad``, clips, not reversible, deterministic, adversarial."""
    attack = cls({"target_model": StubTarget(), "epsilon": EPSILON})
    assert attack.algorithm == algorithm
    assert attack.model_role == "target"
    assert attack.model_methods == ("grad",)
    assert attack._family() == "adversarial"
    assert attack.clips is True
    assert attack.reversible is False
    assert attack.seed is None


def test_resolved_parameters() -> None:
    """Defaults are recorded: L-infinity, 10 steps of a quarter of the budget; ``decay`` for MIM."""
    pgd = ArtPgd({"target_model": StubTarget(), "epsilon": EPSILON})
    mim = ArtMim({"target_model": StubTarget(), "epsilon": EPSILON})
    assert pgd.params == {"epsilon": EPSILON, "norm": "inf", "max_iter": 10, "step_fraction": 0.25}
    assert mim.params == {**pgd.params, "decay": 1.0}


@pytest.mark.parametrize("cls", [ArtPgd, ArtMim])
def test_one_full_step_is_the_local_fgsm(cls: type[ArtPgd], images: np.ndarray) -> None:
    """One step of the whole budget equals ``fgsm_step`` byte for byte."""
    fgsm_step = pytest.importorskip("kcai_data_sampling_fgsm.transformations.fgsm").fgsm_step
    target = StubTarget()
    attack = cls({"target_model": target, "epsilon": EPSILON, "max_iter": 1, "step_fraction": 1.0})
    expected = fgsm_step(images, target.grad(images / 255), EPSILON)
    np.testing.assert_array_equal(attack.apply(images), expected)


@pytest.mark.parametrize("cls", [ArtPgd, ArtMim])
def test_stays_within_the_linf_budget_and_keeps_alpha(cls: type[ArtPgd], images: np.ndarray) -> None:
    """Same shape and dtype, at most 8 grey levels of change, alpha untouched, not a no-op."""
    out = cls({"target_model": StubTarget(), "epsilon": EPSILON}).apply(images)
    assert out.shape == images.shape
    assert out.dtype == np.uint8
    change = np.abs(out.astype(int) - images)
    assert change.max() == round(EPSILON * 255)
    np.testing.assert_array_equal(out[..., 3], images[..., 3])


@pytest.mark.parametrize("cls", [ArtPgd, ArtMim])
def test_stays_within_the_l2_budget(cls: type[ArtPgd], images: np.ndarray) -> None:
    """With ``norm: 2`` the whole perturbation of each image has L2 norm at most ``epsilon`` (plus rounding)."""
    epsilon = 0.5
    out = cls({"target_model": StubTarget(), "epsilon": epsilon, "norm": 2}).apply(images)
    delta = (out.astype(float) - images) / 255
    rounding = np.sqrt(images[0].size) * 0.5 / 255
    norms = np.linalg.norm(delta.reshape(len(images), -1), axis=1)
    assert (norms > 0).all()
    assert (norms <= epsilon + rounding).all()


@pytest.mark.parametrize("cls", [ArtPgd, ArtMim])
def test_independent_of_the_batch(cls: type[ArtPgd], images: np.ndarray) -> None:
    """Attacking the images together or one by one gives the same images."""
    attack = cls({"target_model": StubTarget(), "epsilon": EPSILON})
    alone = np.stack([attack.apply(images[i : i + 1])[0] for i in range(len(images))])
    np.testing.assert_array_equal(attack.apply(images), alone)


def test_refuses_an_unknown_norm() -> None:
    """Only ``inf`` and ``2`` are offered."""
    with pytest.raises(ValueError, match="norm"):
        ArtPgd({"target_model": StubTarget(), "epsilon": EPSILON, "norm": 1})
