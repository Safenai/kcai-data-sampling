"""Time-series end-to-end runs: YAML files in, ledger and ``.npy`` files out.

Every run writes a config to YAML and executes it through
``cli.execute(["-p", ...])``. Assertions inspect only the ledger and payload
artifacts: row counts, parameters, parent IDs, shapes and byte equality.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from kcai_data_sampling_job.cli import execute
from kcai_data_sampling_job.dataloaders.api.timeseries import window_id
from kcai_data_sampling_job.job import GENERATOR_COLUMNS
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from tests.utils.configs import build_config

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
SELECTION = "sensors"
CONSECUTIVE = {"mode": "consecutive", "window": 32, "stride": 16}
#: Windows per series at W = 32, stride 16: 5 + 3 + 3 + 0.
N_WINDOWS = 11
TRANSFORMATIONS = [
    {"type": "low_pass", "cutoff": 0.1},
    {"type": "add_sinusoid", "frequency": 0.25, "amplitude": 1.0},
]
ETT_SELECTION = "ett_ot"
ETT_CONFIGS = ("procedural", "forecasting", "noises", "anomaly")
NINE_ALGORITHMS = {
    "low_pass",
    "add_sinusoid",
    "jitter",
    "drift",
    "level_shift",
    "colored_noise",
    "ou_noise",
    "garch_noise",
    "impulse_noise",
}
EXPECTED_BY_CONFIG = {
    "procedural": {"low_pass", "add_sinusoid", "jitter"},
    "forecasting": {"jitter", "low_pass"},
    "noises": {"colored_noise", "ou_noise", "garch_noise", "impulse_noise"},
    "anomaly": NINE_ALGORITHMS,
}


def _write_yaml(root: Path, config: dict[str, Any], *, name: str = "config.yaml") -> Path:
    """Write a config dict to a YAML file under ``root``."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config))
    return path


def _execute(path: Path) -> None:
    """Run a YAML config exactly as a user would."""
    execute(["-p", str(path)])


def _execute_config(config: dict[str, Any], root: Path, *, name: str = "config.yaml") -> None:
    """Write ``config`` under ``root`` and execute it through the CLI."""
    _execute(_write_yaml(root, config, name=name))


def _loader(path: Path, windows: dict, load_batch_size: int = 64) -> dict:
    return {
        "name": SELECTION,
        "type": "timeseries_parquet",
        "path": str(path),
        "id_column": "series_id",
        "values_column": "values",
        "load_batch_size": load_batch_size,
        "windows": windows,
    }


def _config(
    path: Path,
    root: Path,
    windows: dict | None = None,
    *,
    load: int = 64,
    **kwargs: Any,
) -> dict[str, Any]:
    config = build_config(
        loaders=[_loader(path, windows or CONSECUTIVE, load)],
        output_path=str(root / "ledger" / "{selection}.parquet"),
        samples_dir=str(root / "{selection}"),
        transformations=kwargs.pop("transformations", TRANSFORMATIONS),
        **kwargs,
    )
    config["operations"]["outputs"]["payload_format"] = "npy"
    return config


def _ledger(root: Path, selection: str = SELECTION) -> list[dict]:
    return pq.read_table(root / "ledger" / f"{selection}.parquet").to_pylist()


def _payloads(root: Path, selection: str = SELECTION) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted((root / selection).glob("*.npy"))}


def _ett_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write two synthetic ETT-shaped series (504 hourly steps, 7 channels)."""
    steps, channels = 504, 7
    rng = np.random.default_rng(0)
    hours = np.arange(steps)[:, None]
    values = [30 + 5 * np.sin(2 * np.pi * hours / 24 + k) + rng.normal(size=(steps, channels)) for k in range(2)]
    path = tmp_path_factory.mktemp("ett-data") / "series.parquet"
    pq.write_table(
        pa.table(
            {
                "series_id": ["ETTh1", "ETTh2"],
                "values": [v.tolist() for v in values],
                "start": ["2016-07-01 00:00:00"] * 2,
                "freq": ["1h"] * 2,
            }
        ),
        path,
    )
    return path


def _ett_config(name: str, data: Path, root: Path) -> dict[str, Any]:
    """Load a shipped ETT config and redirect only data/output paths."""
    config = yaml.safe_load((REPO / "examples" / "config" / f"ett-{name}.yaml").read_text())
    config["dataloaders"]["loaders"][0]["path"] = str(data)
    outputs = config["operations"]["outputs"]
    outputs["path"] = str(root / "{selection}" / "ledger.parquet")
    outputs["samples_dir"] = str(root / "{selection}" / "payloads")
    config.setdefault("compute", {})["progress_bar"] = False
    return config


@pytest.fixture(scope="module")
def ett_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _ett_data(tmp_path_factory)


@pytest.fixture(scope="module")
def ett_runs(ett_data: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Run every shipped ETT config once through the YAML-driven CLI."""
    root = tmp_path_factory.mktemp("ett-runs")
    for name in ETT_CONFIGS:
        run_root = root / name
        _execute_config(_ett_config(name, ett_data, run_root), run_root)
    return {name: root / name for name in ETT_CONFIGS}


def test_full_run_writes_one_row_and_one_npy_per_output(series_parquet: Path, tmp_path: Path) -> None:
    """One ledger row and one ``.npy`` file per (window, transformation)."""
    include = ["site", "series_id", "start_window", "end_window"]
    _execute_config(_config(series_parquet, tmp_path, include=include), tmp_path)
    rows = _ledger(tmp_path)
    assert len(rows) == 2 * N_WINDOWS
    assert list(rows[0]) == [*GENERATOR_COLUMNS, *include]
    assert len({r["parent_id"] for r in rows}) == N_WINDOWS

    for row in rows:
        params = json.loads(row["params"])
        assert row["algorithm"] in {"low_pass", "add_sinusoid"}
        if row["algorithm"] == "low_pass":
            assert params["cutoff"] == 0.1
        else:
            assert params["frequency"] == 0.25
            assert params["amplitude"] == 1.0
        assert row["family"] == "procedural"
        assert row["seed"] is None
        assert row["parent_id"] == window_id(row["series_id"], row["start_window"], row["end_window"])
        assert row["end_window"] - row["start_window"] == 31
        payload = np.load(tmp_path / SELECTION / row["artifact"], allow_pickle=False)
        assert payload.shape == (32, 2)
        assert payload.dtype == np.float64

    assert len(_payloads(tmp_path)) == 2 * N_WINDOWS


def test_cutoff_sweep_expands_into_one_transformation_per_value(series_parquet: Path, tmp_path: Path) -> None:
    """``cutoff: {range, samples: 3}`` gives three outputs per window."""
    sweep = [{"type": "low_pass", "cutoff": {"range": [0.05, 0.15], "samples": 3, "mode": "even"}}]
    _execute_config(_config(series_parquet, tmp_path, transformations=sweep), tmp_path)
    rows = _ledger(tmp_path)
    assert len(rows) == 3 * N_WINDOWS
    cutoffs = sorted({json.loads(r["params"])["cutoff"] for r in rows})
    assert np.allclose(cutoffs, [0.05, 0.1, 0.15])


def test_multi_field_sweep_uses_the_cartesian_product(series_parquet: Path, tmp_path: Path) -> None:
    """Two swept fields on one transformation give their product of outputs."""
    sweep = [
        {
            "type": "add_sinusoid",
            "frequency": {"range": [0.05, 0.15], "samples": 3, "mode": "even"},
            "amplitude": {"range": [0.5, 2.0], "samples": 2, "mode": "even"},
        }
    ]
    _execute_config(_config(series_parquet, tmp_path, transformations=sweep), tmp_path)
    rows = _ledger(tmp_path)
    assert len(rows) == N_WINDOWS * 3 * 2
    combos = {
        (round(json.loads(r["params"])["frequency"], 6), round(json.loads(r["params"])["amplitude"], 6)) for r in rows
    }
    assert len(combos) == 6


def test_independent_selection_is_reproducible(series_parquet: Path, tmp_path: Path) -> None:
    """Two runs of a seeded independent selection draw the same windows.

    ``short`` is filtered out: 20 steps hold no 32-step window to draw.
    """
    windows = {"mode": "independent", "window": 32, "n_windows": 2, "non_overlap": True, "seed": 0}
    keep = [{"column": "series_id", "values": ["s0", "s1", "s2"]}]
    for name in ("a", "b"):
        root = tmp_path / name
        config = _config(series_parquet, root, windows)
        config["dataloaders"]["loaders"][0]["filters"] = keep
        _execute_config(config, root)

    assert len(_ledger(tmp_path / "a")) == 3 * 2 * 2
    assert len(_ledger(tmp_path / "b")) == 3 * 2 * 2
    ids = [r["parent_id"] for r in _ledger(tmp_path / "a")]
    assert ids == [r["parent_id"] for r in _ledger(tmp_path / "b")]
    # ``s2`` (64 steps) holds exactly two disjoint 32-step windows.
    assert {window_id("s2", 0, 31), window_id("s2", 32, 63)} <= set(ids)


def test_series_that_cannot_be_windowed_abort_above_the_failure_rate(series_parquet: Path, tmp_path: Path) -> None:
    """1 failed series out of 4 is above the default ``max_failure_rate``."""
    windows = {"mode": "independent", "window": 32, "n_windows": 2, "seed": 0}
    with pytest.raises(RuntimeError, match="1/4 rows failed"):
        _execute_config(_config(series_parquet, tmp_path, windows), tmp_path)


def test_batch_knobs_do_not_change_the_ledger_or_payloads(series_parquet: Path, tmp_path: Path) -> None:
    """Every load / transform / flush combination is byte-identical."""

    def outputs(root: Path) -> tuple[bytes, dict[str, bytes]]:
        ledger = (root / "ledger" / f"{SELECTION}.parquet").read_bytes()
        return ledger, _payloads(root)

    base = tmp_path / "base"
    _execute_config(_config(series_parquet, base, load=64, transform_batch_size=8, flush_batch_size=100), base)
    baseline = outputs(base)
    assert len(baseline[1]) == 2 * N_WINDOWS
    for load, transform, flush in itertools.product((1, 4), (1, 8), (1, 100)):
        root = tmp_path / f"l{load}-t{transform}-f{flush}"
        config = _config(series_parquet, root, load=load, transform_batch_size=transform, flush_batch_size=flush)
        _execute_config(config, root)
        assert outputs(root) == baseline


def test_jitter_is_seeded_and_batch_invariant(series_parquet: Path, tmp_path: Path) -> None:
    """The seed is recorded and decides the noise; the batch knobs never change it."""

    def payloads(root: Path, seed: int = 0, **knobs: int) -> dict[str, bytes]:
        transformations = [{"type": "jitter", "sigma": 0.5, "seed": seed}]
        _execute_config(
            _config(
                series_parquet,
                root,
                transformations=transformations,
                include=["series_id", "start_window", "end_window"],
                **knobs,
            ),
            root,
        )
        return _payloads(root)

    base = tmp_path / "base"
    payloads(base, load=64, transform_batch_size=8)
    baseline = _payloads(base)
    assert len(baseline) == N_WINDOWS
    rows = _ledger(base)
    assert {r["algorithm"] for r in rows} == {"jitter"}
    assert {r["seed"] for r in rows} == {0}
    assert {json.loads(r["params"])["sigma"] for r in rows} == {0.5}
    for row in rows:
        payload = np.load(base / SELECTION / row["artifact"], allow_pickle=False)
        assert payload.shape == (32, 2)
        assert payload.dtype == np.float64
        assert row["parent_id"] == window_id(row["series_id"], row["start_window"], row["end_window"])

    assert payloads(tmp_path / "small", load=1, transform_batch_size=1) == baseline
    assert payloads(tmp_path / "other", seed=1, load=64, transform_batch_size=8) != baseline


def test_selected_channel_is_the_whole_sample(series_parquet: Path, tmp_path: Path) -> None:
    """``channels: [1]`` on the loader: each payload is ``(W, 1)``."""
    config = _config(series_parquet, tmp_path, transformations=[{"type": "low_pass", "cutoff": 0.1}])
    config["dataloaders"]["loaders"][0]["channels"] = [1]
    _execute_config(config, tmp_path)
    rows = _ledger(tmp_path)
    assert len(rows) == N_WINDOWS
    for row in rows:
        payload = np.load(tmp_path / SELECTION / row["artifact"], allow_pickle=False)
        assert payload.shape == (32, 1)
        assert payload.dtype == np.float64


@pytest.mark.parametrize("name", ETT_CONFIGS)
def test_shipped_ett_config_runs_through_yaml(name: str, ett_runs: dict[str, Path]) -> None:
    """Every shipped ETT config runs from YAML on ETT-shaped data."""
    root = ett_runs[name]
    rows = pq.read_table(root / ETT_SELECTION / "ledger.parquet").to_pylist()
    assert {r["algorithm"] for r in rows} == EXPECTED_BY_CONFIG[name]
    assert {r["series_id"] for r in rows} == {"ETTh1", "ETTh2"}
    assert all(r["parent_id"] == window_id(r["series_id"], r["start_window"], r["end_window"]) for r in rows)
    payload = np.load(root / ETT_SELECTION / "payloads" / rows[0]["artifact"], allow_pickle=False)
    assert payload.shape == (168, 1)
    assert payload.dtype == np.float64


def test_all_nine_entry_points_are_reached_through_config(ett_runs: dict[str, Path]) -> None:
    """The union of the shipped ETT ledgers covers every registered algorithm."""
    union = set()
    for name in ETT_CONFIGS:
        table = pq.read_table(ett_runs[name] / ETT_SELECTION / "ledger.parquet", columns=["algorithm"])
        union.update(table["algorithm"].to_pylist())
    assert union == NINE_ALGORITHMS
