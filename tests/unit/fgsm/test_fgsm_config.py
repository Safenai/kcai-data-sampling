"""``FgsmTransformationConfig`` schema behavior.

The adversarial budget ``epsilon`` is required and strictly positive (a value
or an expanding ``SweepConfig`` whose interval stays ``> 0``). Resolution keeps
``params`` exactly ``{epsilon}`` — ``target_model`` excluded from the dump —
and a swept epsilon expands into one concrete instance per value.
"""

import pytest

pytest.importorskip("kcai_data_sampling_fgsm")

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_core.utils.registry import get_transformations_registry, register_model
from kcai_data_sampling_fgsm.api.transformations.fgsm import Fgsm
from kcai_data_sampling_fgsm.configs import FgsmTransformationConfig
from tests.fixtures.registries import StubTarget, registry_snapshot
from tests.utils.configs import build_config, build_loader, build_sweep


def test_epsilon_is_required() -> None:
    """An ``fgsm`` config without ``epsilon`` is refused at load."""
    with pytest.raises(ValueError, match="epsilon\\b"):
        FgsmTransformationConfig.model_validate({"type": "fgsm"})


def test_epsilon_must_be_positive() -> None:
    """A non-positive budget is refused loudly."""
    with pytest.raises(ValueError, match="fgsm.epsilon must be > 0"):
        FgsmTransformationConfig.model_validate({"type": "fgsm", "epsilon": 0.0})
    with pytest.raises(ValueError, match="fgsm.epsilon must be > 0"):
        FgsmTransformationConfig.model_validate({"type": "fgsm", "epsilon": -0.01})


def test_epsilon_sweep_must_stay_positive() -> None:
    """A swept budget whose interval starts at or below zero is refused."""
    with pytest.raises(ValueError, match="fgsm.epsilon sweep .* must stay > 0"):
        FgsmTransformationConfig.model_validate(
            {
                "type": "fgsm",
                "epsilon": {"range": [0.0, 0.1], "samples": 3, "mode": "even"},
            }
        )


def test_resolved_params_are_exactly_epsilon(stub_target: StubTarget) -> None:
    """``target_model`` is excluded; the resolved params are exactly ``{epsilon}``.

    The base consumes ``target_model`` before re-validating the remaining
    parameters against the schema, so the instance's ``params`` carry only the
    budget — the guarantee the row records.
    """
    instance = Fgsm({"target_model": stub_target, "epsilon": 2 / 255})
    assert instance.params == {"epsilon": 2 / 255}
    assert instance.target_model is stub_target


def test_a_swept_epsilon_expands_through_a_job_config(registry_snapshot, tmp_path) -> None:
    """An ``epsilon`` sweep becomes one concrete transformation per value.

    The sweep machinery runs at config validation on the registered schema:
    ``{range: [0.01, 0.03], samples: 3, mode: even}`` on epsilon expands to
    three ``fgsm`` entries, each already a validated, model-bearing instance.
    The stub target is seeded like a notebook would (``register_model``), so
    the ``models:`` section validates in the otherwise model-free env.
    """
    register_model("stub", StubTarget)
    validated = JobConfig.model_validate(
        build_config(
            loaders=[build_loader(parquet_path="missing.parquet")],
            output_path=str(tmp_path / "ledger.parquet"),
            samples_dir=str(tmp_path / "samples"),
            transformations=[
                {"type": "fgsm", "target_model": "stub", "epsilon": build_sweep(0.01, 0.03, 3)}
            ],
            models={"stub": {"type": "stub"}},
        )
    )
    registry = get_transformations_registry()
    resolved = [entry for entry in validated.operations.transformations if entry.type == "fgsm"]
    assert len(resolved) == 3
    assert [entry.epsilon for entry in resolved] == [0.01, 0.02, 0.03]
    assert all(entry.target_model == "stub" for entry in resolved)
    assert registry["fgsm"].Config is FgsmTransformationConfig