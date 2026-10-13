"""The ``npy`` payload writer and its selection by ``outputs.payload_format``.

The writer shares the ``images`` writer's buffering and content-addressed
names; only the file format differs: a ``.npy`` file that loads back to the
exact array, dtype and shape included. The CLI builds whichever registered
writer ``payload_format`` names, keeps ``images`` by default, and refuses an
unknown format or the ledger writer.
"""

from pathlib import Path
import re

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry
from kcai_data_sampling_job.cli import _build_writers
from kcai_data_sampling_job.outputwriter.api.images import ImagesOutputWriter
from kcai_data_sampling_job.outputwriter.api.npy import NpyOutputWriter
from kcai_data_sampling_timeseries.api.selection import TimeSeriesBatch
from kcai_data_sampling_timeseries.api.transformations.low_pass import LowPass
import numpy as np
import pydantic
import pytest

from tests.utils.configs import build_config, build_loader

ARTIFACT_RE = re.compile(r"^(?P<selection>.+)__(?P<id>[0-9a-f]{12})__(?P<c6>[0-9a-f]{6})\.npy$")


def _outputs(batch: TimeSeriesBatch):
    """One Output per window, via a real transformation."""
    return LowPass({"cutoff": 0.1}).transform(batch, batch.value_range)


def _writer(tmp_path: Path, **config) -> NpyOutputWriter:
    return NpyOutputWriter(name="npy", config={"samples_dir": str(tmp_path / "{selection}"), **config})


def test_npy_files_load_back_exactly(tmp_path: Path, synthetic_series_batch: TimeSeriesBatch) -> None:
    """Each file is ``np.load``-able to the output array, dtype and shape kept."""
    writer = _writer(tmp_path, flush_batch_size=100)
    outputs = _outputs(synthetic_series_batch)
    names = [writer.add_payload("series", o) for o in outputs]
    writer.flush()
    for name, output in zip(names, outputs, strict=True):
        assert name is not None
        assert ARTIFACT_RE.match(name)
        loaded = np.load(tmp_path / "series" / name, allow_pickle=False)
        assert loaded.dtype == output.x.dtype
        assert np.array_equal(loaded, output.x)


def test_npy_writer_flushes_at_the_threshold(tmp_path: Path, synthetic_series_batch: TimeSeriesBatch) -> None:
    """Files appear once ``flush_batch_size`` payloads are buffered."""
    writer = _writer(tmp_path, flush_batch_size=2)
    outputs = _outputs(synthetic_series_batch)
    writer.add_payload("series", outputs[0])
    assert not (tmp_path / "series").exists()
    writer.add_payload("series", outputs[1])
    assert len(list((tmp_path / "series").iterdir())) == 2


def test_npy_trace_only_writes_nothing(tmp_path: Path, synthetic_series_batch: TimeSeriesBatch) -> None:
    """``write_samples: false`` returns the name and touches no disk."""
    writer = _writer(tmp_path, write_samples=False)
    name = writer.add_payload("series", _outputs(synthetic_series_batch)[0])
    writer.flush()
    assert name is not None
    assert name.endswith(".npy")
    assert not (tmp_path / "series").exists()


def test_npy_writer_is_registered() -> None:
    """The ``npy`` type resolves to the writer class."""
    assert PluginLoadedRegistry.get_outputwriter_registry()["npy"] is NpyOutputWriter


def _validated(tmp_path: Path, raw_bytes_data: Path, **outputs) -> JobConfig:
    config = build_config(
        loaders=[build_loader(parquet_path=str(raw_bytes_data))],
        output_path=str(tmp_path / "{selection}.parquet"),
        samples_dir=str(tmp_path / "{selection}"),
    )
    config["operations"]["outputs"].update(outputs)
    return JobConfig.model_validate(config)


def test_payload_format_defaults_to_images(tmp_path: Path, raw_bytes_data: Path) -> None:
    """Existing configs, with no ``payload_format``, still write PNG files."""
    payload, ledger = _build_writers(
        _validated(tmp_path, raw_bytes_data), PluginLoadedRegistry.get_outputwriter_registry()
    )
    assert type(payload) is ImagesOutputWriter
    assert ledger is not None


def test_payload_format_npy_builds_the_npy_writer(tmp_path: Path, raw_bytes_data: Path) -> None:
    """``payload_format: npy`` selects the ``npy`` writer."""
    payload, _ = _build_writers(
        _validated(tmp_path, raw_bytes_data, payload_format="npy"),
        PluginLoadedRegistry.get_outputwriter_registry(),
    )
    assert isinstance(payload, NpyOutputWriter)


def test_unknown_payload_format_is_refused(tmp_path: Path, raw_bytes_data: Path) -> None:
    """A format no writer is registered under fails before the run."""
    with pytest.raises(ValueError, match="unknown payload_format 'tiff'"):
        _build_writers(
            _validated(tmp_path, raw_bytes_data, payload_format="tiff"),
            PluginLoadedRegistry.get_outputwriter_registry(),
        )


def test_missing_default_writer_only_warns(tmp_path: Path, raw_bytes_data: Path) -> None:
    """Without an ``images`` writer installed, the default runs trace-only."""
    registry = {k: v for k, v in PluginLoadedRegistry.get_outputwriter_registry().items() if k != "images"}
    payload, _ = _build_writers(_validated(tmp_path, raw_bytes_data), registry)
    assert payload is None


def test_the_ledger_is_not_a_payload_format(tmp_path: Path, raw_bytes_data: Path) -> None:
    """``payload_format: parquet`` is refused at config load."""
    with pytest.raises(pydantic.ValidationError, match="ledger writer"):
        _validated(tmp_path, raw_bytes_data, payload_format="parquet")
