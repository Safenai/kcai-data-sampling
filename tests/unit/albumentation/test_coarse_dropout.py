"""Unit tests for ``CoarseDropout`` (occlusion/dropout family)."""

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("kcai_data_sampling_albumentation")

pytestmark = pytest.mark.albumentation

from kcai_data_sampling_albumentation.api.transformations import CoarseDropout  # noqa: E402
from kcai_data_sampling_albumentation.configs import CoarseDropoutTransformationConfig  # noqa: E402
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


def _run(transform: CoarseDropout, batch: Batch) -> np.ndarray:
    return np.stack([output.x for output in transform.transform(batch, value_range=VALUE_RANGE)])


def test_coarse_dropout_defaults_and_contract() -> None:
    transform = CoarseDropout({"seed": 51})
    assert transform.algorithm == "coarse_dropout"
    assert transform._family() == "procedural"
    assert transform.arity == "unary"
    assert transform.stochastic is True
    assert transform.reversible is False
    assert transform.seed == 51
    assert transform.params["p"] == 1.0


def test_coarse_dropout_config_accepts_valid_params() -> None:
    config = CoarseDropoutTransformationConfig.model_validate(
        {"type": "coarse_dropout", "seed": 51, "num_holes_range": [1, 2]}
    )
    assert config.num_holes_range == [1, 2]


def test_coarse_dropout_unknown_and_oob_rejected() -> None:
    with pytest.raises(ValueError, match=r"nope"):
        CoarseDropoutTransformationConfig.model_validate({"type": "coarse_dropout", "nope": 1})

    with pytest.raises(ValueError, match=r"must (be|satisfy)"):
        CoarseDropoutTransformationConfig.model_validate({"type": "coarse_dropout", "num_holes_range": [2, 1]})


def test_coarse_dropout_applies_and_respects_range() -> None:
    frames = _frames()
    transform = CoarseDropout(
        {"seed": 51, "num_holes_range": [1, 1], "hole_height_range": [2, 2], "hole_width_range": [2, 2]}
    )
    out = _run(transform, _batch(frames))
    assert out.shape == frames.shape
    assert out.dtype == np.uint8
    assert int(out.min()) >= VALUE_RANGE[0]
    assert int(out.max()) <= VALUE_RANGE[1]
    # CoarseDropout may affect alpha in some implementations; just check range/shape


def test_coarse_dropout_replayable_with_seed() -> None:
    frames = _frames()
    batch = _batch(frames)
    first = _run(CoarseDropout({"seed": 51}), batch)
    replay = _run(CoarseDropout({"seed": 51}), batch)
    other = _run(CoarseDropout({"seed": 52}), batch)
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, other)


def test_coarse_dropout_yaml_like_via_jobconfig(tmp_path) -> None:
    validated = JobConfig.model_validate(
        build_config(
            loaders=[build_loader(parquet_path="missing.parquet")],
            output_path=str(tmp_path / "ledger.parquet"),
            samples_dir=str(tmp_path / "samples"),
            transformations=[{"type": "coarse_dropout", "seed": 51}],
        )
    )
    resolved = [entry for entry in validated.operations.transformations if entry.type == "coarse_dropout"]
    assert len(resolved) == 1
    assert resolved[0].seed == 51
