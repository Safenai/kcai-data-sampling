"""E2E smoke: all 6 albumentations families via YAML with p tested on one transform."""

import json
from pathlib import Path

from kcai_data_sampling_job.cli import run
from kcai_data_sampling_job.job import GENERATOR_COLUMNS
import pyarrow.parquet as pq
import pytest

from tests.utils.configs import build_loader

pytest.importorskip("kcai_data_sampling_albumentation.api.transformations.gaussian_blur")

pytestmark = pytest.mark.albumentation


def _config(root: Path, raw_bytes_data: Path) -> dict:
    config = {
        "dataloaders": {"loaders": [build_loader(parquet_path=raw_bytes_data)]},
        "operations": {
            "transformations": [
                {"name": "hsv", "type": "hue_saturation_value", "seed": 21},
                {"name": "blur", "type": "gaussian_blur", "seed": 11, "p": 0.5},
                {"name": "noise", "type": "gauss_noise", "seed": 31},
                {"name": "affine", "type": "affine", "seed": 41},
                {"name": "dropout", "type": "coarse_dropout", "seed": 51},
                {"name": "rain", "type": "random_rain", "seed": 61},
            ],
            "outputs": {
                "path": str(root / "ledger" / "{selection}.parquet"),
                "samples_dir": str(root / "{selection}"),
                "include": ["height", "width"],
            },
        },
    }
    return config


def test_albumentations_families_yaml_smoke(raw_bytes_data, tmp_path) -> None:
    root = tmp_path / "alb_families"
    cfg = _config(root, raw_bytes_data)
    summary = run(cfg)
    assert summary == {"synthetic": 48}

    table = pq.read_table(str(root / "ledger" / "synthetic.parquet"))
    assert table.column_names == list(GENERATOR_COLUMNS) + ["height", "width"]
    assert table.num_rows == 48

    algorithms = table.column("algorithm").to_pylist()
    from collections import Counter

    assert Counter(algorithms) == {
        "hue_saturation_value": 8,
        "gaussian_blur": 8,
        "gauss_noise": 8,
        "affine": 8,
        "coarse_dropout": 8,
        "random_rain": 8,
    }
    assert set(algorithms) == {
        "hue_saturation_value",
        "gaussian_blur",
        "gauss_noise",
        "affine",
        "coarse_dropout",
        "random_rain",
    }

    assert table.column("family").to_pylist() == ["procedural"] * 48
    assert table.column("arity").to_pylist() == ["unary"] * 48
    assert table.column("reversible").to_pylist() == [False] * 48
    assert table.column("tool_model").to_pylist() == [None] * 48
    assert table.column("target_model").to_pylist() == [None] * 48

    params_list = [json.loads(p) for p in table.column("params").to_pylist()]
    for algo, params in zip(algorithms, params_list, strict=True):
        if algo == "gaussian_blur":
            assert params["p"] == 0.5
        else:
            assert params["p"] == 1.0

    payloads = list((root / "synthetic").glob("*.png"))
    assert len(payloads) == 48
