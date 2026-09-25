"""Plugin wiring of the ``parquet`` dataloader.

The core never imports the loader plugin directly: at config load the registry
maps the entry's ``type`` to the registered class and validates the entry
against that class's own schema, and the loader's decode mode is refused at
construction when not implemented. The wiring ends with a real selection
yielding one decoded ``(B, H, W, 4)`` batch per chunk, image column dropped.
"""

import numpy as np
import pydantic
import pytest

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry
from tests.utils.configs import build_config, build_loader


def test_parquet_loader_is_registered_under_the_core_surface() -> None:
    """The ``parquet`` type resolves to the plugin class in the core registry.

    The core's only grip on the loader is the registry entry (the split
    surface); asserting the identity plus the attached ``Config`` schema proves
    the plugin is wired, not copied.
    """
    from kcai_data_sampling_job.dataloaders.api.parquet import (
        ParquetDataLoader,
        ParquetImageLoaderConfig,
    )

    registry = PluginLoadedRegistry.get_dataloaders_registry()
    assert "parquet" in registry
    assert registry["parquet"] is ParquetDataLoader
    assert registry["parquet"].Config is ParquetImageLoaderConfig


def test_config_resolves_loader_entries_into_the_plugin_schema(
    raw_bytes_data, tmp_path
) -> None:
    """A config dict becomes a ``ParquetImageLoaderConfig`` at validation.

    Loading a job config resolves each ``type`` entry against the registry and
    validates it with the loader's own schema, so a valid bytes-column entry
    carries its datatype-specific keys (name, path, sample_path, id_column,
    decode).
    """
    validated = JobConfig.model_validate(
        build_config(
            loaders=[build_loader(parquet_path=str(raw_bytes_data))],
            output_path=str(tmp_path / "{selection}.parquet"),
            samples_dir=str(tmp_path / "{selection}"),
        )
    )
    entry = validated.dataloaders.loaders[0]
    from kcai_data_sampling_job.dataloaders.api.parquet import ParquetImageLoaderConfig

    assert isinstance(entry, ParquetImageLoaderConfig)
    assert entry.type == "parquet"
    assert entry.name == "synthetic"
    assert entry.sample_path[0].column == "img"
    assert entry.id_column == "id"
    assert entry.decode == "img_bytes"


def test_unknown_loader_type_is_refused(tmp_path) -> None:
    """A loader ``type`` outside the registry fails loudly at config load.

    Refusing at validation (not at run) keeps a mistyped loader from reaching
    the job: the error is raised by ``JobConfig.model_validate``.
    """
    loader = build_loader(parquet_path="does-not-matter.parquet")
    loader["type"] = "csv"
    config = build_config(
        loaders=[loader],
        output_path=str(tmp_path / "l.parquet"),
        samples_dir=str(tmp_path / "s"),
    )
    with pytest.raises(ValueError, match="unknown dataloader type 'csv'"):
        JobConfig.model_validate(config)


def test_plugin_schema_refuses_unknown_keys(tmp_path) -> None:
    """Keys the parquet schema does not know are rejected, not ignored.

    ``extra="forbid"`` on the registered schema surfaces a stray key as a
    validation error instead of silently carrying an unused knob into the job.
    """
    loader = build_loader(parquet_path="does-not-matter.parquet")
    loader["not_a_key"] = 1
    config = build_config(
        loaders=[loader],
        output_path=str(tmp_path / "l.parquet"),
        samples_dir=str(tmp_path / "s"),
    )
    with pytest.raises(pydantic.ValidationError, match="Extra inputs are not permitted"):
        JobConfig.model_validate(config)


def test_unimplemented_decode_mode_is_refused() -> None:
    """``decode: rgba`` passes the schema but is refused at construction.

    The schema admits the value for forward compatibility; the loader is where
    "not implemented yet" bites — a loud, local failure.
    """
    from kcai_data_sampling_job.dataloaders.api.parquet import (
        ParquetDataLoader,
        ParquetImageLoaderConfig,
    )

    config = ParquetImageLoaderConfig(
        name="synthetic",
        path="no.parquet",
        sample_path=[{"column": "img"}],
        decode="rgba",
    )
    with pytest.raises(ValueError, match="not implemented yet"):
        ParquetDataLoader(name="synthetic", config=config)


def test_selection_decodes_one_bytes_column_per_chunk(synthetic_selection) -> None:
    """The wired selection yields one decoded batch with the image column gone.

    bootstrapped iteration over the synthetic parquet gives exactly one batch
    (8 rows ≤ load_batch_size): a ``(8, 32, 32, 4)`` uint8 stack whose ids are
    the ``id`` column values, while the remaining source columns stay arrow-
    native and the ``img`` column never reaches the batch.
    """
    synthetic_selection.bootstrap(None)
    batches = list(synthetic_selection)
    assert len(batches) == 1
    batch = batches[0]
    assert batch.name == "synthetic"
    assert batch.dataset == "synthetic"
    assert batch.ids == [f"syn_{i:04d}" for i in range(8)]
    assert batch.data.shape == (8, 32, 32, 4)
    assert batch.data.dtype == np.uint8
    assert batch.sample_axes == ("height", "width", "channel")
    assert batch.value_range == (0.0, 255.0)
    assert "img" not in batch.columns.column_names
    assert set(batch.columns.column_names) == {"id", "height", "width", "source"}