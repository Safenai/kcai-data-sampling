"""The ``Fgsm`` transformation class: declarations, apply, and output checks.

``Fgsm`` wraps the normalized sign-direction step into the unary contract: a
``target`` model role with the ``grad`` method set, ``clips`` declared (the
epsilon step may push pixels past the selection's range), non-reversible,
deterministic (``seed=None``), and an ``adversarial`` family derived from the
target role. ``apply`` runs through the ``TransformationRunner`` with the stub
target: row fields are recorded, outputs stay uint8 in range, and pixels the
step does not touch are byte-identical. ``check_output`` refuses a
model-graded output that breaks the numeric contract (wrong shape /
non-finite) before the step runs.
"""

from kcai_data_sampling_core.utils.runner import TransformationRunner
import numpy as np
import pytest
from tests.fixtures.registries import StubTarget

pytest.importorskip("kcai_data_sampling_fgsm")

pytestmark = pytest.mark.fgsm

from kcai_data_sampling_fgsm.api.transformations.fgsm import Fgsm  # noqa: E402


@pytest.fixture
def fgsm(stub_target: StubTarget) -> Fgsm:
    """An ``Fgsm`` instance stepping along the stub target's gradient.

    Returns:
        An ``Fgsm`` at ``epsilon = 2/255`` against ``stub_target``.
    """
    return Fgsm({"target_model": stub_target, "epsilon": 2 / 255})


def test_declarations_adversarial_target_clips_and_deterministic(fgsm: Fgsm) -> None:
    """The class declares the adversarial contract on every row.

    ``family`` derives ``"adversarial"`` from the ``target`` role, the model
    method set is ``grad``, ``clips`` lets the step overshoot the value range,
    the map is not reversible, and the deterministic step records no seed.
    """
    assert fgsm.algorithm == "fgsm"
    assert fgsm.model_role == "target"
    assert fgsm.model_methods == ("grad",)
    assert fgsm._family() == "adversarial"
    assert fgsm.clips is True
    assert fgsm.reversible is False
    assert fgsm.stochastic is False
    assert fgsm.seed is None
    assert fgsm.arity == "unary"


def test_apply_records_row_fields_and_stays_in_range(fgsm: Fgsm, synthetic_batch) -> None:
    """``apply`` returns uint8 outputs in the value range with row facts.

    Running through the runner — the real transformation path — emits one
    ``Output`` per row carrying the adversarial declarations: the stub's name
    in ``target_model``, the derived ``adversarial`` family, no seed, and the
    resolved ``epsilon`` as the only parameter.
    """
    outputs = TransformationRunner(synthetic_batch).run(fgsm)
    assert len(outputs) == len(synthetic_batch)
    for output in outputs:
        assert output.x.dtype == np.uint8
        assert output.x.shape == synthetic_batch.images[0].shape
        assert int(output.x.min()) >= 0
        assert int(output.x.max()) <= 255
        assert output.family == "adversarial"
        assert output.arity == "unary"
        assert output.target_model == "stub"
        assert output.seed is None
        assert output.params == {"epsilon": 2 / 255}


def test_apply_moves_only_the_sign_pixels_by_the_exact_delta(fgsm: Fgsm, synthetic_batch) -> None:
    """Pixels the stub's sign does not touch stay byte-identical; the others move.

    The stub's gradient is +1 on the red plane where the green-blue mean is
    dark, 0 elsewhere — so a pixel either moves by exactly ``round(2/255 * 255)
    == 2`` on red (unless clipped) or stays byte-identical everywhere else.
    """
    for row, image in enumerate(synthetic_batch.images):
        normalized = image.astype(np.float64) / 255.0
        dark = normalized[..., 1:3].mean(axis=-1) < 0.5
        sign = np.zeros(image.shape, dtype=np.float64)
        sign[..., 0][dark] = 1.0
        output = TransformationRunner(synthetic_batch).run(fgsm)[row].x
        moved = sign > 0
        assert np.array_equal(output[~moved], image[~moved])
        expected = np.clip(image.astype(np.int16) + np.rint(2 * sign), 0, 255).astype(np.uint8)
        assert np.array_equal(output[moved], expected[moved])


def test_check_output_refuses_a_wrong_shaped_grad(fgsm: Fgsm) -> None:
    """A badly-shaped ``grad`` is refused loudly, before any step.

    ``check_output`` runs inside ``apply`` ahead of ``fgsm_step``; the error
    names the model, the method and the expected shape.
    """

    class _WrongShape:
        name = "wrong_shape"

        def grad(self, xs: np.ndarray) -> np.ndarray:
            return np.zeros((xs.shape[0],), dtype=np.float64)

    broken = Fgsm({"target_model": _WrongShape(), "epsilon": 2 / 255})
    with pytest.raises(ValueError, match=r"wrong_shape\.grad returned"):
        broken.apply(
            np.zeros((2, 4, 4, 3), dtype=np.uint8),
            fgsm.rngs(["a", "b"]),
        )


def test_check_output_refuses_a_non_finite_grad(fgsm: Fgsm) -> None:
    """A ``grad`` leaking ``nan``/``inf`` is refused loudly, before any step.

    Finiteness is the numeric contract of a model output; the refusal names the
    model and the method, and the batch is left untouched.
    """

    class _NaN:
        name = "nan_grad"

        def grad(self, xs: np.ndarray) -> np.ndarray:
            return np.full(xs.shape, np.nan, dtype=np.float64)

    broken = Fgsm({"target_model": _NaN(), "epsilon": 2 / 255})
    with pytest.raises(ValueError, match=r"nan_grad\.grad returned non-finite"):
        broken.apply(
            np.zeros((2, 4, 4, 3), dtype=np.uint8),
            fgsm.rngs(["a", "b"]),
        )
