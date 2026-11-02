"""Shared tests for application probability ``p`` on albumentations transforms."""

import pytest

pytest.importorskip("kcai_data_sampling_albumentation")

pytestmark = pytest.mark.albumentation

from kcai_data_sampling_albumentation.api.transformations import GaussianBlur  # noqa: E402
from kcai_data_sampling_albumentation.configs import GaussianBlurTransformationConfig  # noqa: E402
from kcai_data_sampling_core.models.config import JobConfig  # noqa: E402
from tests.utils.configs import build_config, build_loader, build_sweep  # noqa: E402


def test_p_defaults_to_one_on_concrete_transform() -> None:
    transform = GaussianBlur({"seed": 11})
    assert transform.params["p"] == 1.0


def test_p_can_be_set_to_fixed_value() -> None:
    transform = GaussianBlur({"seed": 11, "p": 0.5})
    assert transform.params["p"] == 0.5


def test_p_accepts_sweep_via_jobconfig(tmp_path) -> None:
    validated = JobConfig.model_validate(
        build_config(
            loaders=[build_loader(parquet_path="missing.parquet")],
            output_path=str(tmp_path / "ledger.parquet"),
            samples_dir=str(tmp_path / "samples"),
            transformations=[{"type": "gaussian_blur", "seed": 11, "p": build_sweep(0.25, 1.0, 3)}],
        )
    )
    resolved = [entry for entry in validated.operations.transformations if entry.type == "gaussian_blur"]
    values = sorted([entry.p for entry in resolved])
    assert values[0] == 0.25
    assert values[2] == 1.0
    assert len(values) == 3


def test_p_out_of_bounds_rejected() -> None:
    with pytest.raises(ValueError, match=r"must (be|satisfy)"):
        GaussianBlurTransformationConfig.model_validate({"type": "gaussian_blur", "p": 1.5})

    with pytest.raises(ValueError, match=r"must (be|satisfy)"):
        GaussianBlurTransformationConfig.model_validate({"type": "gaussian_blur", "p": -0.1})


def test_p_is_part_of_resolved_params() -> None:
    transform = GaussianBlur({"seed": 11, "p": 0.7})
    assert "p" in transform.params
    assert transform.params["p"] == 0.7
