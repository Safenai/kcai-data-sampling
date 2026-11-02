"""Unit tests for ``HueSaturationValue`` (color/tone family)."""

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("kcai_data_sampling_albumentation")

pytestmark = pytest.mark.albumentation

from kcai_data_sampling_albumentation.api.transformations import HueSaturationValue  # noqa: E402
from kcai_data_sampling_albumentation.configs import HueSaturationValueTransformationConfig  # noqa: E402
from kcai_data_sampling_core.api.selection import Batch  # noqa: E402
from kcai_data_sampling_core.models.config import JobConfig  # noqa: E402
from tests.utils.configs import build_config, build_loader  # noqa: E402

VALUE_RANGE = (0, 255)


def _frames(rows: int = 3, size: int = 8, channels: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(rows, size, size, channels), dtype=np.uint8)


def _batch(frames: np.ndarray, ids: list[str] | None = None) -> Batch:
    row_ids = ids if ids is not None else [f"r{i}" for i in range(len(frames))]
    return Batch(
        name="albumentation",
        dataset="unit",
        ids=row_ids,
        columns=pa.table({"tag": row_ids}),
        data=frames,
    )


def _run(transform: HueSaturationValue, batch: Batch) -> np.ndarray:
    return np.stack([output.x for output in transform.transform(batch, value_range=VALUE_RANGE)])


def test_hue_saturation_value_defaults_and_contract() -> None:
    transform = HueSaturationValue({"seed": 21})
    assert transform.algorithm == "hue_saturation_value"
    assert transform._family() == "procedural"
    assert transform.arity == "unary"
    assert transform.stochastic is True
    assert transform.reversible is False
    assert transform.seed == 21
    assert transform.params["p"] == 1.0


def test_hue_saturation_value_config_accepts_valid_params() -> None:
    config = HueSaturationValueTransformationConfig.model_validate(
        {"type": "hue_saturation_value", "seed": 21, "hue_shift_limit": [-20, 20], "sat_shift_limit": [-30, 30]}
    )
    assert config.hue_shift_limit == [-20, 20]
    assert config.sat_shift_limit == [-30, 30]


def test_hue_saturation_value_unknown_and_oob_rejected() -> None:
    with pytest.raises(ValueError, match=r"nope"):
        HueSaturationValueTransformationConfig.model_validate({"type": "hue_saturation_value", "nope": 1})

    with pytest.raises(ValueError, match=r"must (be|satisfy)"):
        HueSaturationValueTransformationConfig.model_validate(
            {"type": "hue_saturation_value", "hue_shift_limit": [50, 10]}
        )


def test_hue_saturation_value_applies_and_respects_range() -> None:
    frames = _frames()
    transform = HueSaturationValue({"seed": 21, "hue_shift_limit": [10, 10], "sat_shift_limit": [0, 0]})
    out = _run(transform, _batch(frames))
    assert out.shape == frames.shape
    assert out.dtype == np.uint8
    assert int(out.min()) >= VALUE_RANGE[0]
    assert int(out.max()) <= VALUE_RANGE[1]
    assert np.array_equal(out[..., 3], frames[..., 3])


def test_hue_saturation_value_replayable_with_seed() -> None:
    frames = _frames()
    batch = _batch(frames)
    first = _run(HueSaturationValue({"seed": 21}), batch)
    replay = _run(HueSaturationValue({"seed": 21}), batch)
    other = _run(HueSaturationValue({"seed": 22}), batch)
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, other)


def test_hue_saturation_value_yaml_like_via_jobconfig(tmp_path) -> None:
    validated = JobConfig.model_validate(
        build_config(
            loaders=[build_loader(parquet_path="missing.parquet")],
            output_path=str(tmp_path / "ledger.parquet"),
            samples_dir=str(tmp_path / "samples"),
            transformations=[{"type": "hue_saturation_value", "seed": 21}],
        )
    )
    resolved = [entry for entry in validated.operations.transformations if entry.type == "hue_saturation_value"]
    assert len(resolved) == 1
    assert resolved[0].seed == 21
