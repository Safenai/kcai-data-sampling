"""Config fixtures for full job runs.

Small parametrized wrappers over ``tests.utils.configs.build_config``: e2e
tests only ever vary a focused set of things — the include/
exclude passthrough, trace-only mode, the swept ``fraction``, and the three
image-column forms — so each config here is one focused scenario builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.fixtures.data import PATH_COLUMN
from tests.utils.configs import build_config, build_loader, build_sweep


def standard_config(
    root: Path,
    raw_bytes_data: Path,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    write_samples: bool = True,
    compute_seed: int | None = 42,
) -> dict[str, Any]:
    """The nominal full-run config over the raw-bytes synthetic sidecar.

    One loader (the bytes form), both deterministic algorithms, outputs under
    ``root``. Called with ``include=["height", "width"]`` by the full-run
    inventory to exercise passthrough.

    Args:
        root: Output root for the run.
        raw_bytes_data: The raw-bytes parquet fixture path.
        include: Source columns passed through to the ledger.
        exclude: Source columns withheld from the ledger (wins over include).
        write_samples: Emit payload files, or run trace-only.
        compute_seed: Seed for ``compute.seed``.

    Returns:
        A config dict ready for ``cli.run``.
    """
    return build_config(
        loaders=[build_loader(parquet_path=raw_bytes_data)],
        output_root=root,
        include=include,
        exclude=exclude,
        write_samples=write_samples,
        compute_seed=compute_seed,
    )


def swept_fraction_config(
    root: Path,
    raw_bytes_data: Path,
    lo: float = 0.2,
    hi: float = 0.8,
    samples: int = 3,
) -> dict[str, Any]:
    """A config whose ``crop_resize.fraction`` is an even sweep.

    Args:
        root: Output root for the run.
        raw_bytes_data: The raw-bytes parquet fixture path.
        lo: Sweep lower bound.
        hi: Sweep upper bound.
        samples: Number of concrete fractions the sweep expands into.

    Returns:
        The swept config dict.
    """
    return build_config(
        loaders=[build_loader(parquet_path=raw_bytes_data)],
        output_root=root,
        transformations=[
            {"type": "horizontal_flip"},
            {"type": "crop_resize", "fraction": build_sweep(lo, hi, samples)},
        ],
    )


def all_forms_config(root: Path, raw_bytes_data: Path, path_data: dict[str, dict[str, str]]) -> dict[str, Any]:
    """One run whose three loaders serve the same frames each way.

    The bytes loader plus the relative-path loader (resolved against
    ``prefix``) and the absolute-path loader share one config, so the run's
    three selections must agree on rows and pixels.

    Args:
        root: Output root for the run.
        raw_bytes_data: The raw-bytes parquet fixture path.
        path_data: The relative/absolute path fixtures.

    Returns:
        The three-loader config dict.
    """
    relative = path_data["relative"]
    absolute = path_data["absolute"]
    return build_config(
        loaders=[
            build_loader(parquet_path=raw_bytes_data, name="bytes"),
            build_loader(
                parquet_path=relative["parquet"],
                name="relative",
                prefix=relative["prefix"],
                column=PATH_COLUMN,
                decode=None,
            ),
            build_loader(
                parquet_path=absolute["parquet"],
                name="absolute",
                column=PATH_COLUMN,
                decode=None,
            ),
        ],
        output_path=str(root / "ledger" / "{selection}.parquet"),
        samples_dir=str(root / "{selection}"),
    )


def write_yaml(path: Path, config: dict[str, Any]) -> Path:
    """Persist a config dict as YAML for a CLI-driven run.

    Args:
        path: Destination YAML path.
        config: The config dict.

    Returns:
        The written path.
    """
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return path
