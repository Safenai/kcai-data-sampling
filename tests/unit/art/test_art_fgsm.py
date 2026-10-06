"""The ``ArtFgsm`` transformation: ART's FGSM through the ``TargetEstimator`` adapter.

The reference is the local ``fgsm_step``: with the same target gradient and
budget, ART's ``FastGradientMethod`` (L-infinity, one step, clipped to
``[0, 1]``) must produce byte-identical batches. The run goes through the
``TransformationRunner`` with the stub target, so the row facts are checked on
the real path too.
"""

from kcai_data_sampling_core.utils.runner import TransformationRunner
import numpy as np
import pytest
from tests.fixtures.registries import StubTarget

pytest.importorskip("kcai_data_sampling_art")

pytestmark = pytest.mark.art

from kcai_data_sampling_art.api.transformations.fgsm import ArtFgsm  # noqa: E402

EPSILON = 2 / 255


@pytest.fixture
def art_fgsm(stub_target: StubTarget) -> ArtFgsm:
    """An ``ArtFgsm`` instance at ``epsilon = 2/255`` against ``stub_target``.

    Returns:
        The transformation.
    """
    return ArtFgsm({"target_model": stub_target, "epsilon": EPSILON})


def test_declarations_match_the_local_fgsm(art_fgsm: ArtFgsm) -> None:
    """Target role on ``grad``, clips, not reversible, deterministic, adversarial."""
    assert art_fgsm.algorithm == "art_fgsm"
    assert art_fgsm.model_role == "target"
    assert art_fgsm.model_methods == ("grad",)
    assert art_fgsm._family() == "adversarial"
    assert art_fgsm.clips is True
    assert art_fgsm.reversible is False
    assert art_fgsm.seed is None


def test_apply_matches_the_local_fgsm_step(art_fgsm: ArtFgsm, stub_target: StubTarget, synthetic_batch) -> None:
    """ART's FGSM equals ``fgsm_step`` byte for byte, row facts recorded."""
    fgsm_step = pytest.importorskip("kcai_data_sampling_fgsm.transformations.fgsm").fgsm_step

    outputs = TransformationRunner(synthetic_batch).run(art_fgsm)
    assert len(outputs) == len(synthetic_batch)
    for image, output in zip(synthetic_batch.images, outputs, strict=True):
        xs = image[None]
        expected = fgsm_step(xs, stub_target.grad(xs / np.iinfo(xs.dtype).max), EPSILON)[0]
        assert output.x.dtype == np.uint8
        np.testing.assert_array_equal(output.x, expected)
        assert output.target_model == "stub"
        assert output.family == "adversarial"
        assert output.params == {"epsilon": EPSILON}
