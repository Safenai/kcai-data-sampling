"""The ``timeseries_parquet`` loader: series rows in, window batches out.

A small table of three univariate series (20, 7 and 3 steps) is cut into
windows by both selections. Checked: which windows exist and their ids, that
the draw of independent windows is seeded and never depends on the batch
size, that overlap is refused when asked, how a bad series counts, and that
every batch declares the window space and carries the source columns.
"""

from itertools import pairwise
from pathlib import Path

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry
from kcai_data_sampling_job.dataloaders.api.timeseries import (
    ConsecutiveTimeSeriesSamples,
    IndependentTimeSeriesSamples,
    IndependentWindowsConfig,
    TimeSeriesLoaderConfig,
    TimeSeriesParquetDataLoader,
    independent_starts,
)
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
import pytest

from tests.utils.configs import build_config

LENGTHS = {"a": 20, "b": 7, "c": 3}


def _write_series(path: Path, rows: dict[str, list] | None = None) -> Path:
    """Write one series per row: ``values[t] = 100 * series_index + t``.

    The values encode their own position, so a window's content tells which
    series and which steps it was cut from.
    """
    if rows is None:
        rows = {sid: [100.0 * i + t for t in range(n)] for i, (sid, n) in enumerate(LENGTHS.items())}
    table = pa.table({"series_id": list(rows), "values": list(rows.values()), "site": ["x"] * len(rows)})
    pq.write_table(table, path)
    return path


def _loader_config(path: Path, windows: dict, **extra) -> TimeSeriesLoaderConfig:
    """A resolved loader config over ``path``."""
    entry = {
        "name": "sensors",
        "type": "timeseries_parquet",
        "path": str(path),
        "id_column": "series_id",
        "values_column": "values",
        "windows": windows,
        **extra,
    }
    return TimeSeriesLoaderConfig.model_validate(entry)


def _batches(path: Path, windows: dict, errors: dict | None = None, **extra) -> list:
    """Every batch of the loader's single selection."""
    loader = TimeSeriesParquetDataLoader("sensors", _loader_config(path, windows, **extra), errors=errors)
    (selection,) = loader.get_selections()
    selection.bootstrap(None)
    return list(selection)


def _ids(batches: list) -> list[str]:
    return [i for b in batches for i in b.ids]


def test_loader_is_registered() -> None:
    """The ``timeseries_parquet`` type resolves to the plugin and its schema."""
    registry = PluginLoadedRegistry.get_dataloaders_registry()
    assert registry["timeseries_parquet"] is TimeSeriesParquetDataLoader
    assert registry["timeseries_parquet"].Config is TimeSeriesLoaderConfig


def test_job_config_resolves_the_entry(tmp_path: Path) -> None:
    """A job config validates the entry against the loader's own schema."""
    loader = {
        "name": "sensors",
        "type": "timeseries_parquet",
        "path": str(tmp_path / "s.parquet"),
        "values_column": "values",
        "windows": {"mode": "independent", "window": 4, "n_windows": 2},
    }
    config = build_config(
        loaders=[loader],
        output_path=str(tmp_path / "{selection}.parquet"),
        samples_dir=str(tmp_path / "{selection}"),
    )
    entry = JobConfig.model_validate(config).dataloaders.loaders[0]
    assert isinstance(entry, TimeSeriesLoaderConfig)
    assert isinstance(entry.windows, IndependentWindowsConfig)


def test_consecutive_windows_and_ids(tmp_path: Path) -> None:
    """W = 4, stride 3: starts 0, 3, …, 15 on ``a``, 0 and 3 on ``b``, none on ``c``.

    A window's id is ``<series>@<last step>``, and its values are the steps it
    covers.
    """
    batches = _batches(_write_series(tmp_path / "s.parquet"), {"mode": "consecutive", "window": 4, "stride": 3})
    assert _ids(batches) == [f"a@{s + 3}" for s in range(0, 17, 3)] + ["b@3", "b@6"]
    data = np.concatenate([b.data for b in batches])
    assert data.shape == (8, 4, 1)
    assert np.array_equal(data[1, :, 0], [3.0, 4.0, 5.0, 6.0])
    assert np.array_equal(data[6, :, 0], [100.0, 101.0, 102.0, 103.0])


def test_stride_defaults_to_the_window(tmp_path: Path) -> None:
    """Without ``stride``, windows tile the series: 5 on ``a``, 1 on ``b``."""
    batches = _batches(_write_series(tmp_path / "s.parquet"), {"mode": "consecutive", "window": 4})
    assert _ids(batches) == ["a@3", "a@7", "a@11", "a@15", "a@19", "b@3"]


def test_batches_declare_the_window_space_and_carry_columns(tmp_path: Path) -> None:
    """Each batch is ``(B, W, C)`` float64, time/channel, no range, plus columns."""
    batches = _batches(_write_series(tmp_path / "s.parquet"), {"mode": "consecutive", "window": 4}, load_batch_size=4)
    assert [len(b) for b in batches] == [4, 2]
    first = batches[0]
    assert first.data.dtype == np.float64
    assert first.sample_axes == ("time", "channel")
    assert first.value_range is None
    assert first.columns.column_names == ["series_id", "site", "window_start", "window_end"]
    assert first.columns.column("window_start").to_pylist() == [0, 4, 8, 12]
    assert first.columns.column("window_end").to_pylist() == [3, 7, 11, 15]


def test_batch_size_never_changes_the_windows(tmp_path: Path) -> None:
    """Any ``load_batch_size`` yields the same windows in the same order."""
    path = _write_series(tmp_path / "s.parquet")
    windows = {"mode": "independent", "window": 3, "n_windows": 2, "seed": 7}
    reference = _batches(path, windows, load_batch_size=1000)
    for size in (1, 2, 5):
        batches = _batches(path, windows, load_batch_size=size)
        assert _ids(batches) == _ids(reference)
        assert np.array_equal(np.concatenate([b.data for b in batches]), np.concatenate([b.data for b in reference]))


def test_independent_windows_are_seeded(tmp_path: Path) -> None:
    """N windows per series, ``<series>@<start>`` ids, same seed → same draw."""
    path = _write_series(tmp_path / "s.parquet")
    windows = {"mode": "independent", "window": 3, "n_windows": 2, "seed": 1}
    ids = _ids(_batches(path, windows, errors={"on_decode_failure": "silent_fail"}))
    assert ids == _ids(_batches(path, windows))
    assert [i.split("@")[0] for i in ids] == ["a", "a", "b", "b", "c"][: len(ids)]
    assert ids != _ids(_batches(path, {**windows, "seed": 2}))


def test_non_overlapping_draw_keeps_windows_apart() -> None:
    """With ``non_overlap``, sorted starts are at least W apart, for any seed."""
    for seed in range(50):
        config = IndependentWindowsConfig(window=5, n_windows=4, non_overlap=True, seed=seed)
        starts = independent_starts(23, config, "s")
        assert all(b - a >= 5 for a, b in pairwise(starts))
        assert starts[-1] + 5 <= 23


def test_draw_that_cannot_fit_is_refused() -> None:
    """Three disjoint windows of 4 steps do not fit in 11 steps."""
    config = IndependentWindowsConfig(window=4, n_windows=3, non_overlap=True)
    with pytest.raises(ValueError, match="do not fit"):
        independent_starts(11, config, "s")


def test_series_that_cannot_be_windowed_counts_as_a_failure(tmp_path: Path) -> None:
    """``c`` (3 steps) cannot hold 2 distinct 3-step windows: skipped, counted."""
    path = _write_series(tmp_path / "s.parquet")
    loader = TimeSeriesParquetDataLoader(
        "sensors", _loader_config(path, {"mode": "independent", "window": 3, "n_windows": 2})
    )
    (selection,) = loader.get_selections()
    selection.bootstrap(None)
    ids = _ids(list(selection))
    assert {i.split("@")[0] for i in ids} == {"a", "b"}
    assert selection.failures == 1
    assert selection.nb_samples == 3


def test_fail_fast_raises_on_a_bad_series(tmp_path: Path) -> None:
    """Under ``fail_fast`` the same series aborts the read."""
    path = _write_series(tmp_path / "s.parquet")
    with pytest.raises(ValueError, match="do not fit"):
        _batches(path, {"mode": "independent", "window": 3, "n_windows": 2}, errors={"on_decode_failure": "fail_fast"})


def test_multichannel_series(tmp_path: Path) -> None:
    """``list<list<float>>`` rows give ``(W, C)`` windows."""
    rows = {"m": [[float(t), -float(t)] for t in range(6)]}
    batches = _batches(_write_series(tmp_path / "s.parquet", rows), {"mode": "consecutive", "window": 3})
    assert batches[0].data.shape == (2, 3, 2)
    assert np.array_equal(batches[0].data[1, :, 1], [-3.0, -4.0, -5.0])


def test_channel_count_must_not_change(tmp_path: Path) -> None:
    """A series with another channel count than the first is a failure."""
    rows = {"one": [[0.0, 1.0]] * 4, "two": [[0.0, 1.0, 2.0]] * 4}
    path = _write_series(tmp_path / "s.parquet", rows)
    loader = TimeSeriesParquetDataLoader("sensors", _loader_config(path, {"mode": "consecutive", "window": 2}))
    (selection,) = loader.get_selections()
    selection.bootstrap(None)
    assert _ids(list(selection)) == ["one@1", "one@3"]
    assert selection.failures == 1


def test_null_series_is_a_failure(tmp_path: Path) -> None:
    """A null values row is skipped and counted."""
    table = pa.table({"series_id": ["a", "b"], "values": [[1.0, 2.0], None]})
    path = tmp_path / "s.parquet"
    pq.write_table(table, path)
    loader = TimeSeriesParquetDataLoader("sensors", _loader_config(path, {"mode": "consecutive", "window": 2}))
    (selection,) = loader.get_selections()
    selection.bootstrap(None)
    assert _ids(list(selection)) == ["a@1"]
    assert selection.failures == 1


def test_filters_select_series(tmp_path: Path) -> None:
    """A ``filters`` entry keeps only the matching series."""
    path = _write_series(tmp_path / "s.parquet")
    batches = _batches(path, {"mode": "consecutive", "window": 4}, filters=[{"column": "series_id", "values": ["b"]}])
    assert _ids(batches) == ["b@3"]


def test_ids_fall_back_to_the_row_index(tmp_path: Path) -> None:
    """Without ``id_column`` the series id is its row index."""
    path = _write_series(tmp_path / "s.parquet")
    config = _loader_config(path, {"mode": "consecutive", "window": 7}).model_copy(update={"id_column": None})
    (selection,) = TimeSeriesParquetDataLoader("sensors", config).get_selections()
    selection.bootstrap(None)
    assert _ids(list(selection)) == ["0@6", "0@13", "1@6"]


def test_selection_class_follows_the_mode(tmp_path: Path) -> None:
    """``consecutive`` and ``independent`` build their own selection class."""
    path = tmp_path / "s.parquet"
    consecutive = TimeSeriesParquetDataLoader("s", _loader_config(path, {"mode": "consecutive", "window": 2}))
    independent = TimeSeriesParquetDataLoader(
        "s", _loader_config(path, {"mode": "independent", "window": 2, "n_windows": 1})
    )
    assert isinstance(consecutive.get_selections()[0], ConsecutiveTimeSeriesSamples)
    assert isinstance(independent.get_selections()[0], IndependentTimeSeriesSamples)


def test_batch_count_estimate(tmp_path: Path) -> None:
    """The progress estimate counts the windows of each mode."""
    path = _write_series(tmp_path / "s.parquet")
    for windows, expected in (
        ({"mode": "consecutive", "window": 4, "stride": 3}, 3),
        ({"mode": "independent", "window": 2, "n_windows": 2}, 2),
    ):
        loader = TimeSeriesParquetDataLoader("s", _loader_config(path, windows, load_batch_size=3))
        (selection,) = loader.get_selections()
        selection.bootstrap(None)
        assert selection.get_nb_batches() == expected


def test_iteration_before_bootstrap_yields_nothing(tmp_path: Path) -> None:
    """A selection that was never bootstrapped has no dataset to read."""
    loader = TimeSeriesParquetDataLoader(
        "s", _loader_config(tmp_path / "s.parquet", {"mode": "consecutive", "window": 2})
    )
    (selection,) = loader.get_selections()
    assert list(selection) == []
    assert "consecutive" in repr(selection)


@pytest.mark.parametrize(
    "windows",
    [
        {"mode": "consecutive", "window": 0},
        {"mode": "consecutive", "window": 4, "stride": 0},
        {"mode": "independent", "window": 4},
        {"mode": "sliding", "window": 4},
        {"mode": "consecutive", "window": 4, "n_windows": 2},
    ],
)
def test_invalid_windows_are_refused(tmp_path: Path, windows: dict) -> None:
    """Bad lengths, a missing count, an unknown mode or a foreign key are refused."""
    with pytest.raises(pydantic.ValidationError):
        _loader_config(tmp_path / "s.parquet", windows)


@pytest.mark.parametrize(
    "extra",
    [
        {"sample_path": [{"column": "values"}]},
        {"transform": [{"column": "site", "to_type": "str"}]},
        {"sampling_rate": 0},
    ],
)
def test_unsupported_keys_are_refused(tmp_path: Path, extra: dict) -> None:
    """Image-only keys and a non-positive sampling rate are refused."""
    with pytest.raises(pydantic.ValidationError):
        _loader_config(tmp_path / "s.parquet", {"mode": "consecutive", "window": 2}, **extra)


def test_values_that_are_not_a_series_are_refused() -> None:
    """A 3-D values row is not a series."""
    from kcai_data_sampling_job.dataloaders.api.timeseries import as_series

    with pytest.raises(ValueError, match="dimensions"):
        as_series([[[1.0]]])
