"""The metadata-only ledger columns of a job run.

One row per output: the 13 generator columns are always present and always
fixed in shape (deterministic ⇒ ``seed is None``; ``params`` a sorted JSON
string; ``artifact`` the content-addressed payload name), plus any pass-through
source columns selected by ``include``/``exclude`` — with ``exclude`` winning
and generator columns never overwritten. Runs go through the real
``SamplingJob`` with the registered loader and writers.
"""

import json
import re

import numpy as np
import pyarrow.parquet as pq

from kcai_data_sampling_images.api.transformations.crop_resize import CropResize
from kcai_data_sampling_images.api.transformations.horizontal_flip import HorizontalFlip
from kcai_data_sampling_job.dataloaders.api.parquet import (
    ParquetDataLoader,
    ParquetImageLoaderConfig,
)
from kcai_data_sampling_job.job import GENERATOR_COLUMNS, SamplingJob
from kcai_data_sampling_job.outputwriter.api.images import ImagesOutputWriter
from kcai_data_sampling_job.outputwriter.api.parquet import ParquetOutputWriter


#: The 13 generator columns, in the order the job emits them.
assert GENERATOR_COLUMNS == (
    "selection",
    "dataloader",
    "parent_id",
    "id",
    "algorithm",
    "family",
    "arity",
    "reversible",
    "params",
    "seed",
    "tool_model",
    "target_model",
    "artifact",
)


def _run(tmp_path, data_path, *, include=None, exclude=None, write_samples=True):
    """Run a real job over the synthetic selection; return summary and ledger table.

    The job is assembled from the same objects the CLI builds: a
    ``ParquetDataLoader`` over the bytes fixture, the two deterministic image
    algorithms, and the images + parquet writers mapped onto ``tmp_path``.
    """
    loader = ParquetDataLoader(
        name="synthetic",
        config=ParquetImageLoaderConfig(
            name="synthetic",
            path=str(data_path),
            sample_path=[{"column": "img"}],
            id_column="id",
            load_batch_size=8,
        ),
    )
    job = SamplingJob(
        dataloaders={"synthetic": loader},
        transformations=[HorizontalFlip({}), CropResize({"fraction": 0.5})],
        payload_writer=ImagesOutputWriter(
            name="images",
            config={"samples_dir": str(tmp_path / "{selection}"), "write_samples": write_samples},
        ),
        ledger_writer=ParquetOutputWriter(
            name="ledger",
            config={"path_pattern": str(tmp_path / "ledger" / "{selection}.parquet")},
        ),
        progress_bar=False,
        include_columns=include,
        exclude_columns=exclude,
    )
    summary = job.run()
    table = pq.read_table(str(tmp_path / "ledger" / "synthetic.parquet"))
    return summary, table


def test_metadata_only_ledger_carries_the_thirteen_generator_columns(
    raw_bytes_data, tmp_path
) -> None:
    """A plain run writes exactly the 13 generator columns — no pixels, no extras.

    With no ``include``/``exclude`` nothing passes through, so the ledger is
    metadata-only: 8 samples × 2 algorithms = 16 rows and not a single source
    column in sight.
    """
    summary, table = _run(tmp_path, raw_bytes_data)
    assert summary == {"synthetic": 16}
    assert table.column_names == list(GENERATOR_COLUMNS)
    assert table.num_rows == 16


def test_generator_columns_hold_the_row_facts(raw_bytes_data, tmp_path) -> None:
    """The generator columns encode each output's identity and declarations.

    Row-by-row: ``parent_id`` is a source sample, ``seed`` records ``None``
    for deterministic algorithms, ``params`` is the resolved, sorted JSON, and
    every output is declared ``procedural``/``unary`` with no model behind it.
    """
    _, table = _run(tmp_path, raw_bytes_data)
    assert table.column("selection").to_pylist() == ["synthetic"] * 16
    assert table.column("dataloader").to_pylist() == ["synthetic"] * 16
    assert all(
        pid in {f"syn_{i:04d}" for i in range(8)} for pid in table.column("parent_id").to_pylist()
    )
    assert len(set(table.column("id").to_pylist())) == 16
    assert table.column("algorithm").to_pylist() == ["horizontal_flip", "crop_resize"] * 8
    assert table.column("family").to_pylist() == ["procedural"] * 16
    assert table.column("arity").to_pylist() == ["unary"] * 16
    assert table.column("reversible").to_pylist() == [True, False] * 8
    assert table.column("seed").to_pylist() == [None] * 16
    assert table.column("tool_model").to_pylist() == [None] * 16
    assert table.column("target_model").to_pylist() == [None] * 16

    params = [json.loads(v) for v in table.column("params").to_pylist()]
    assert params == [
        {},
        {"fraction": 0.5, "top": 0, "left": 0},
    ] * 8


def test_include_passes_the_named_source_columns_through(
    raw_bytes_data, tmp_path
) -> None:
    """``include`` appends the selected source columns after the generators.

    The requested height/width values survive the round trip into the ledger,
    beside the 13 generator columns.
    """
    _, table = _run(tmp_path, raw_bytes_data, include=["height", "width"])
    assert table.column_names == list(GENERATOR_COLUMNS) + ["height", "width"]
    assert table.column("height").to_pylist() == [32] * 16
    assert table.column("width").to_pylist() == [32] * 16


def test_include_wildcards_and_generator_protection(raw_bytes_data, tmp_path) -> None:
    """A ``*`` include pulls source columns but never a generator column.

    ``id`` is a generator column, so it must not be re-appended as a
    pass-through; height/width/source do make it through.
    """
    _, table = _run(tmp_path, raw_bytes_data, include=["*"])
    assert set(table.column_names) == set(GENERATOR_COLUMNS) | {"height", "width", "source"}
    assert "source" in table.column_names


def test_exclude_wins_over_include(raw_bytes_data, tmp_path) -> None:
    """``exclude`` withholds a source column even under a wildcard include."""
    _, table = _run(tmp_path, raw_bytes_data, include=["*"], exclude=["source"])
    assert set(table.column_names) == set(GENERATOR_COLUMNS) | {"height", "width"}


def test_every_row_names_a_real_payload_artifact(raw_bytes_data, tmp_path) -> None:
    """The ``artifact`` column points at an existing hashed PNG per row.

    In writer mode each row names its payload file, which exists on disk under
    the selection directory and matches the content-addressed naming.
    """
    summary, table = _run(tmp_path, raw_bytes_data)
    artifacts = table.column("artifact").drop_null().to_pylist()
    assert len(artifacts) == summary["synthetic"]
    payload_dir = tmp_path / "synthetic"
    for name in artifacts:
        assert (payload_dir / name).exists()
        assert re.fullmatch(
            r"synthetic__[0-9a-f]{12}__[0-9a-f]{6}\.png",
            name,
        ) is not None


def test_trace_only_run_names_artifacts_but_writes_no_files(
    raw_bytes_data, tmp_path
) -> None:
    """``write_samples: false`` keeps the recipe trace without touching the disk.

    A pure recipe trace: same 16 ledger rows, each naming its content-addressed
    artifact, but no payload directory appears — the hashes stand in for samples
    a re-run would materialize.
    """
    _, table = _run(tmp_path, raw_bytes_data, write_samples=False)
    artifacts = table.column("artifact").to_pylist()
    assert len(artifacts) == 16
    assert all(
        re.fullmatch(r"synthetic__[0-9a-f]{12}__[0-9a-f]{6}\.png", name) is not None
        for name in artifacts
    )
    assert not (tmp_path / "synthetic").exists()


def test_zero_transformations_is_a_clean_noop(raw_bytes_data, tmp_path) -> None:
    """An empty recipe runs to a zero-row summary instead of crashing.

    A selection with no transformations (e.g. a job venv without the images
    algorithms installed) is a valid no-op: every batch decodes, the loop
    emits nothing, and no ledger file appears — the chunk loop must not
    divide by a zero chunk (``range()`` step) on an empty recipe.
    """
    loader = ParquetDataLoader(
        name="synthetic",
        config=ParquetImageLoaderConfig(
            name="synthetic",
            path=str(raw_bytes_data),
            sample_path=[{"column": "img"}],
            id_column="id",
            load_batch_size=8,
        ),
    )
    job = SamplingJob(
        dataloaders={"synthetic": loader},
        transformations=[],
        payload_writer=ImagesOutputWriter(
            name="images",
            config={"samples_dir": str(tmp_path / "{selection}"), "write_samples": True},
        ),
        ledger_writer=ParquetOutputWriter(
            name="ledger",
            config={"path_pattern": str(tmp_path / "ledger" / "{selection}.parquet")},
        ),
        progress_bar=False,
    )
    assert job.run() == {"synthetic": 0}
    assert not (tmp_path / "ledger" / "synthetic.parquet").exists()
    assert not (tmp_path / "synthetic").exists()