"""Unit tests for ``GaussNoise`` (noise family)."""

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("kcai_data_sampling_albumentation")

pytestmark = pytest.mark.albumentation

from kcai_data_sampling_albumentation.api.transformations import GaussNoise  # noqa: E402
from kcai_data_sampling_albumentation.configs import GaussNoiseTransformationConfig  # noqa: E402
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


def _run(transform: GaussNoise, batch: Batch) -> np.ndarray:
    return np.stack([output.x for output in transform.transform(batch, value_range=VALUE_RANGE)])


def test_gauss_noise_defaults_and_contract() -> None:
    transform = GaussNoise({"seed": 31})
    assert transform.algorithm == "gauss_noise"
    assert transform._family() == "procedural"
    assert transform.arity == "unary"
    assert transform.stochastic is True
    assert transform.reversible is False
    assert transform.seed == 31
    assert transform.params["p"] == 1.0


def test_gauss_noise_config_accepts_valid_params() -> None:
    config = GaussNoiseTransformationConfig.model_validate({"type": "gauss_noise", "seed": 31, "std_range": [0.1, 0.5]})
    assert config.std_range == [0.1, 0.5]


def test_gauss_noise_unknown_and_oob_rejected() -> None:
    with pytest.raises(ValueError, match=r"nope"):
        GaussNoiseTransformationConfig.model_validate({"type": "gauss_noise", "nope": 1})

    with pytest.raises(ValueError, match=r"must (be|satisfy)"):
        GaussNoiseTransformationConfig.model_validate({"type": "gauss_noise", "std_range": [0.5, 0.1]})


def test_gauss_noise_applies_and_respects_range() -> None:
    frames = _frames()
    transform = GaussNoise({"seed": 31, "std_range": [0.2, 0.2]})
    out = _run(transform, _batch(frames))
    assert out.shape == frames.shape
    assert out.dtype == np.uint8
    assert int(out.min()) >= VALUE_RANGE[0]
    assert int(out.max()) <= VALUE_RANGE[1]
    assert np.array_equal(out[..., 3], frames[..., 3])


def test_gauss_noise_replayable_with_seed() -> None:
    frames = _frames()
    batch = _batch(frames)
    first = _run(GaussNoise({"seed": 31}), batch)
    replay = _run(GaussNoise({"seed": 31}), batch)
    other = _run(GaussNoise({"seed": 32}), batch)
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, other)


def test_gauss_noise_yaml_like_via_jobconfig(tmp_path) -> None:
    validated = JobConfig.model_validate(
        build_config(
            loaders=[build_loader(parquet_path="missing.parquet")],
            output_path=str(tmp_path / "ledger.parquet"),
            samples_dir=str(tmp_path / "samples"),
            transformations=[{"type": "gauss_noise", "seed": 31}],
        )
    )
    resolved = [entry for entry in validated.operations.transformations if entry.type == "gauss_noise"]
    assert len(resolved) == 1
    assert resolved[0].seed == 31
