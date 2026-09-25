"""Core + job smoke: parquet decode + the job CLI, no ``-images``.

Runs inside a ``core+job`` venv: proves ``-job`` installs its own registry
surface (the ``parquet`` dataloader and the ``parquet``/``images`` writers) and
drives a minimal parquet + YAML through the CLI. ``-images`` is not installed
here, so the config deliberately declares **no** transformations: the run
finishes with zero rows but the loader's decode path is exercised directly.
"""

import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry
from kcai_data_sampling_job.cli import execute
from kcai_data_sampling_job.dataloaders.api.parquet import (
    ParquetDataLoader,
    ParquetImageLoaderConfig,
)


def _frames(count: int = 3, size: int = 8) -> list[bytes]:
    """Distinct two-tone RGBA frames as raw bytes (no Pillow needed)."""
    out = []
    for k in range(count):
        frame = np.zeros((size, size, 4), dtype=np.uint8)
        frame[:, : size // 2, :3] = 100 + 60 * k
        frame[:, size // 2 :, :3] = 40
        frame[:, :, 3] = 255
        out.append(frame.tobytes())
    return out


def main() -> int:
    scratch = pathlib.Path(sys.argv[1])
    scratch.mkdir(parents=True, exist_ok=True)
    registry = PluginLoadedRegistry.get_dataloaders_registry()
    assert "parquet" in registry, "job must register its dataloader plugin"
    writers = PluginLoadedRegistry.get_outputwriter_registry()
    assert {"images", "parquet"} <= set(writers), "job must register its writer plugins"

    ids = [f"r{i}" for i in range(3)]
    pq.write_table(
        pa.table(
            {
                "img": pa.array(_frames(), type=pa.binary()),
                "id": ids,
                "height": pa.array([8] * 3, type=pa.uint16()),
                "width": pa.array([8] * 3, type=pa.uint16()),
            }
        ),
        scratch / "raw.parquet",
    )

    loader = ParquetDataLoader(
        name="smoke",
        config=ParquetImageLoaderConfig(
            name="smoke",
            path=str(scratch / "raw.parquet"),
            sample_path=[{"column": "img"}],
            id_column="id",
            load_batch_size=3,
        ),
    )
    selection = loader.get_selections()[0]
    selection.bootstrap(None)
    batches = list(selection)
    assert len(batches) == 1
    assert batches[0].data.shape == (3, 8, 8, 4)
    assert batches[0].data.dtype == np.uint8

    yaml_path = scratch / "job.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "compute:",
                "  progress_bar: false",
                "dataloaders:",
                "  loaders:",
                "    - type: parquet",
                "      name: smoke",
                f"      path: {scratch / 'raw.parquet'}",
                "      sample_path:",
                "        - column: img",
                "      id_column: id",
                "      load_batch_size: 3",
                "      decode: img_bytes",
                "operations:",
                "  transformations: []",
                "  outputs:",
                f"    path: {scratch}/outputs/ledger/{{selection}}.parquet",
                f"    samples_dir: {scratch}/outputs/{{selection}}",
                "    write_samples: true",
                "    flush_batch_size: 1",
                "",
            ]
        )
    )
    execute(["-p", str(yaml_path)])
    print("smoke_job ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())