"""A time-series job, end to end: series rows in, ledger and ``.npy`` files out.

Every run is the full CLI→job→writer path over the ``series_parquet`` fixture
(four 2-channel series of 100, 70, 64 and 20 steps), cut into 32-step windows.
Checked: one ledger row and one ``.npy`` file per (window, transformation),
each file equal to the algorithm applied to its parent window, sweeps, the
independent selection, the seeded ``jitter``, and byte-identical outputs
whatever the batch knobs.
The shipped ETT example config is run too, on a table with ETT's schema.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from kcai_data_sampling_job.cli import run
from kcai_data_sampling_job.dataloaders.api.timeseries import (
    TimeSeriesLoaderConfig,
    TimeSeriesParquetDataLoader,
    window_id,
)
from kcai_data_sampling_job.job import GENERATOR_COLUMNS
from kcai_data_sampling_timeseries.transformations import add_sinusoid, low_pass
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from tests.utils.configs import build_config

REPO = Path(__file__).resolve().parents[2]
SELECTION = "sensors"
CONSECUTIVE = {"mode": "consecutive", "window": 32, "stride": 16}
#: Windows per series at W = 32, stride 16: 5 + 3 + 3 + 0.
N_WINDOWS = 11
TRANSFORMATIONS = [
    {"type": "low_pass", "cutoff": 0.1},
    {"type": "add_sinusoid", "frequency": 0.25, "amplitude": 1.0},
]


def _loader(path: Path, windows: dict, load_batch_size: int = 64) -> dict:
    return {
        "name": SELECTION,
        "type": "timeseries_parquet",
        "path": str(path),
        "id_column": "series_id",
        "values_column": "values",
        "sampling_rate": 50,
        "load_batch_size": load_batch_size,
        "windows": windows,
    }


def _config(path: Path, root: Path, windows: dict | None = None, *, load: int = 64, **kwargs) -> dict:
    config = build_config(
        loaders=[_loader(path, windows or CONSECUTIVE, load)],
        output_path=str(root / "ledger" / "{selection}.parquet"),
        samples_dir=str(root / "{selection}"),
        transformations=kwargs.pop("transformations", TRANSFORMATIONS),
        **kwargs,
    )
    config["operations"]["outputs"]["payload_format"] = "npy"
    return config


def _ledger(root: Path) -> list[dict]:
    return pq.read_table(root / "ledger" / f"{SELECTION}.parquet").to_pylist()


def _windows_by_id(path: Path, windows: dict) -> dict[str, np.ndarray]:
    """Every window of the selection, by id, read straight from the loader."""
    config = TimeSeriesLoaderConfig.model_validate(_loader(path, windows))
    (selection,) = TimeSeriesParquetDataLoader(SELECTION, config).get_selections()
    selection.bootstrap(None)
    return {i: x for batch in selection for i, x in zip(batch.ids, batch.data, strict=True)}


def test_full_run_writes_one_row_and_one_npy_per_output(series_parquet: Path, tmp_path: Path) -> None:
    """Each ``.npy`` file is its algorithm applied to its parent window."""
    include = ["site", "series_id", "start_window", "end_window"]
    summary = run(_config(series_parquet, tmp_path, include=include))
    assert summary == {SELECTION: 2 * N_WINDOWS}

    rows = _ledger(tmp_path)
    assert list(rows[0]) == [*GENERATOR_COLUMNS, *include]
    parents = _windows_by_id(series_parquet, CONSECUTIVE)
    assert {r["parent_id"] for r in rows} == set(parents)
    for row in rows:
        x = parents[row["parent_id"]]
        params = json.loads(row["params"])
        if row["algorithm"] == "low_pass":
            expected = low_pass(x[None], params["cutoff"])[0]
        else:
            expected = add_sinusoid(x[None], params["frequency"], params["amplitude"], params["phase"])[0]
        payload = np.load(tmp_path / SELECTION / row["artifact"], allow_pickle=False)
        assert payload.shape == (32, 2)
        assert payload.dtype == np.float64
        assert np.array_equal(payload, expected)
        assert row["family"] == "procedural"
        assert row["seed"] is None
        assert row["parent_id"] == window_id(row["series_id"], row["start_window"], row["end_window"])
        assert row["end_window"] - row["start_window"] == 31


def test_cutoff_sweep_expands_into_one_transformation_per_value(series_parquet: Path, tmp_path: Path) -> None:
    """``cutoff: {range, samples: 3}`` gives three outputs per window."""
    sweep = [{"type": "low_pass", "cutoff": {"range": [0.05, 0.15], "samples": 3, "mode": "even"}}]
    assert run(_config(series_parquet, tmp_path, transformations=sweep)) == {SELECTION: 3 * N_WINDOWS}
    cutoffs = sorted({json.loads(r["params"])["cutoff"] for r in _ledger(tmp_path)})
    assert np.allclose(cutoffs, [0.05, 0.1, 0.15])


def test_independent_selection_is_reproducible(series_parquet: Path, tmp_path: Path) -> None:
    """Two runs of a seeded independent selection draw the same windows.

    ``short`` is filtered out: 20 steps hold no 32-step window to draw.
    """
    windows = {"mode": "independent", "window": 32, "n_windows": 2, "non_overlap": True, "seed": 0}
    keep = [{"column": "series_id", "values": ["s0", "s1", "s2"]}]
    summaries = []
    for name in ("a", "b"):
        config = _config(series_parquet, tmp_path / name, windows)
        config["dataloaders"]["loaders"][0]["filters"] = keep
        summaries.append(run(config))
    assert summaries == [{SELECTION: 3 * 2 * 2}] * 2
    ids = [r["parent_id"] for r in _ledger(tmp_path / "a")]
    assert ids == [r["parent_id"] for r in _ledger(tmp_path / "b")]
    # ``s2`` (64 steps) holds exactly two disjoint 32-step windows.
    assert {window_id("s2", 0, 31), window_id("s2", 32, 63)} <= set(ids)


def test_series_that_cannot_be_windowed_abort_above_the_failure_rate(series_parquet: Path, tmp_path: Path) -> None:
    """1 failed series out of 4 is above the default ``max_failure_rate``."""
    windows = {"mode": "independent", "window": 32, "n_windows": 2, "seed": 0}
    with pytest.raises(RuntimeError, match="1/4 rows failed"):
        run(_config(series_parquet, tmp_path, windows))


def test_batch_knobs_do_not_change_the_ledger_or_payloads(series_parquet: Path, tmp_path: Path) -> None:
    """Every load / transform / flush combination is byte-identical."""

    def outputs(root: Path) -> tuple[bytes, dict[str, bytes]]:
        ledger = (root / "ledger" / f"{SELECTION}.parquet").read_bytes()
        return ledger, {p.name: p.read_bytes() for p in sorted((root / SELECTION).glob("*.npy"))}

    run(_config(series_parquet, tmp_path / "base", load=64, transform_batch_size=8, flush_batch_size=100))
    baseline = outputs(tmp_path / "base")
    assert len(baseline[1]) == 2 * N_WINDOWS
    for load, transform, flush in itertools.product((1, 4), (1, 8), (1, 100)):
        root = tmp_path / f"l{load}-t{transform}-f{flush}"
        run(_config(series_parquet, root, load=load, transform_batch_size=transform, flush_batch_size=flush))
        assert outputs(root) == baseline


def test_jitter_is_seeded_and_batch_invariant(series_parquet: Path, tmp_path: Path) -> None:
    """The seed is recorded and decides the noise; the batch knobs never change it.

    Each payload minus its parent window is noise of the requested scale, two
    runs with the same seed are byte-identical whatever the batch sizes, and
    another seed gives other payloads.
    """
    noise = [{"type": "jitter", "sigma": 0.5, "seed": 0}]

    def payloads(root: Path, seed: int = 0, **knobs: int) -> dict[str, bytes]:
        transformations = [{**noise[0], "seed": seed}]
        run(_config(series_parquet, root, transformations=transformations, **knobs))
        return {p.name: p.read_bytes() for p in sorted((root / SELECTION).glob("*.npy"))}

    baseline = payloads(tmp_path / "base", load=64, transform_batch_size=8)
    assert len(baseline) == N_WINDOWS
    rows = _ledger(tmp_path / "base")
    assert {r["seed"] for r in rows} == {0}
    parents = _windows_by_id(series_parquet, CONSECUTIVE)
    deltas = np.stack([np.load(tmp_path / "base" / SELECTION / r["artifact"]) - parents[r["parent_id"]] for r in rows])
    assert abs(deltas.std() - 0.5) < 0.05
    assert payloads(tmp_path / "small", load=1, transform_batch_size=1) == baseline
    assert payloads(tmp_path / "other", seed=1, load=64, transform_batch_size=8) != baseline


def test_selected_channel_is_the_whole_sample(series_parquet: Path, tmp_path: Path) -> None:
    """``channels: [1]`` on the loader: each payload is ``(W, 1)``, the filtered channel 1."""
    config = _config(series_parquet, tmp_path, transformations=[{"type": "low_pass", "cutoff": 0.1}])
    config["dataloaders"]["loaders"][0]["channels"] = [1]
    assert run(config) == {SELECTION: N_WINDOWS}
    parents = _windows_by_id(series_parquet, CONSECUTIVE)
    for row in _ledger(tmp_path):
        payload = np.load(tmp_path / SELECTION / row["artifact"])
        assert payload.shape == (32, 1)
        assert np.array_equal(payload, low_pass(parents[row["parent_id"]][None, :, [1]], 0.1)[0])


def test_shipped_example_runs(tmp_path: Path) -> None:
    """``ett-procedural.yaml`` runs as shipped, on a table with ETT's schema.

    ETT itself is downloaded by ``scripts/fetch_ett_sample.py`` and may not be
    redistributed, so the test writes two series with the same columns
    (``series_id``, 7-channel ``values``, ``start``, ``freq``) and 504 hourly
    steps each. Only the data and output paths are moved under ``tmp_path``:
    one-week windows every 84 hours give 5 windows per series, and the config
    asks for 3 + 1 + 1 outputs per window (the low-pass sweep, the sinusoid,
    the seeded jitter), on the oil temperature (channel 6) only.
    """
    steps, channels = 504, 7
    rng = np.random.default_rng(0)
    hours = np.arange(steps)[:, None]
    values = [30 + 5 * np.sin(2 * np.pi * hours / 24 + k) + rng.normal(size=(steps, channels)) for k in range(2)]
    data = tmp_path / "series.parquet"
    pq.write_table(
        pa.table(
            {
                "series_id": ["ETTh1", "ETTh2"],
                "values": [v.tolist() for v in values],
                "start": ["2016-07-01 00:00:00"] * 2,
                "freq": ["1h"] * 2,
            }
        ),
        data,
    )
    config = yaml.safe_load((REPO / "examples" / "config" / "ett-procedural.yaml").read_text())
    config["dataloaders"]["loaders"][0]["path"] = str(data)
    outputs = config["operations"]["outputs"]
    outputs["path"] = str(tmp_path / "out" / "{selection}" / "ledger.parquet")
    outputs["samples_dir"] = str(tmp_path / "out" / "{selection}" / "payloads")
    config["compute"] = {"progress_bar": False}
    assert run(config) == {"ett_ot": 2 * 5 * 5}
    rows = pq.read_table(tmp_path / "out" / "ett_ot" / "ledger.parquet").to_pylist()
    assert {r["series_id"] for r in rows} == {"ETTh1", "ETTh2"}
    assert {r["seed"] for r in rows if r["algorithm"] == "jitter"} == {0}
    assert all(r["parent_id"] == window_id(r["series_id"], r["start_window"], r["end_window"]) for r in rows)
    payload = np.load(tmp_path / "out" / "ett_ot" / "payloads" / rows[0]["artifact"])
    assert payload.shape == (168, 1)
