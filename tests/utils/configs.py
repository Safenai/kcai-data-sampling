"""Inline builders for job configs.

Tests assemble their YAML config dicts here instead of shipping checked-in
``generated/`` trees: one explicit ``build_config`` entry point that
defaults the transformation list and outputs shape, plus small helpers for the
loader entries (all three column forms) and sweep dictionaries. Every returned
dict is a plain nested dict matching the ``JobConfig`` schema the job CLI
validates, so a test can hand it to the CLI, or mutate it for an edge case,
with no further machinery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.fixtures.data import (
    ID_COLUMN,
    IMG_COLUMN,
    PATH_COLUMN,
    SELECTION_NAME,
    SYNTHETIC_ROWS,
)

DEFAULT_COMPUTE_SEED = 42


def build_loader(
    *,
    parquet_path: str | Path,
    name: str = SELECTION_NAME,
    column: str = IMG_COLUMN,
    prefix: str | None = None,
    id_column: str | None = ID_COLUMN,
    load_batch_size: int = SYNTHETIC_ROWS,
    decode: str | None = "img_bytes",
    **extra: Any,
) -> dict[str, Any]:
    """Build one ``parquet`` loader entry dict for the config's dataloaders.

    The entry carries the plugin-owned keys the loader registry resolves:
    ``type: parquet``, the ``sample_path`` single-column object (with an
    optional ``prefix`` for relative paths), the ``id_column``, the
    ``load_batch_size`` knob, and ``decode: img_bytes`` for the raw-bytes
    form. For the path column forms pass ``column=PATH_COLUMN`` and drop
    ``decode`` (path rows carry their shape in the file).

    Args:
        parquet_path: Sidecar parquet file or dataset directory.
        name: Unique loader name; also the selection name (one selection per
            loader).
        column: Image column: ``IMG_COLUMN`` (raw bytes) or ``PATH_COLUMN``.
        prefix: Optional ``sample_path.prefix`` resolving relative paths.
        id_column: Row-identifier column, or ``None`` for row-index ids.
        load_batch_size: Rows pulled from the selection per batch.
        decode: Payload form; ``"img_bytes"`` for bytes columns, ``None`` for
            path columns.
        **extra: Any further keys merged into the entry (e.g. ``image_shape``).

    Returns:
        A loader dict ready for ``build_config(loaders=[...])``.
    """
    entry: dict[str, Any] = {
        "type": "parquet",
        "name": name,
        "path": str(parquet_path),
        "sample_path": [{"column": column, "prefix": prefix}],
    }
    if id_column is not None:
        entry["id_column"] = id_column
    if load_batch_size is not None:
        entry["load_batch_size"] = load_batch_size
    if decode is not None:
        entry["decode"] = decode
    entry.update(extra)
    return entry


def build_sweep(
    lo: float,
    hi: float,
    samples: int,
    mode: str = "even",
) -> dict[str, Any]:
    """Build a ``SweepConfig`` dictionary for a swept parameter.

    Args:
        lo: Sweep interval lower bound (algorithm-enforced, e.g. ``> 0``).
        hi: Sweep interval upper bound (algorithm-enforced, e.g. ``<= 1``).
        samples: Number of concrete values the sweep expands into.
        mode: ``"even"`` (evenly spaced) or ``"random"`` (drawn from the
            compute seed).

    Returns:
        The sweep dict, e.g. ``{"range": [0.2, 0.8], "samples": 3,
        "mode": "even"}``, valid as a ``crop_resize.fraction`` value.
    """
    return {"range": [lo, hi], "samples": samples, "mode": mode}


def build_config(
    *,
    loaders: list[dict[str, Any]],
    transformations: list[dict[str, Any]] | None = None,
    transform_batch_size: int | None = None,
    output_root: str | Path | None = None,
    output_path: str | None = None,
    samples_dir: str | None = None,
    write_samples: bool = True,
    flush_batch_size: int = 100,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    compute_seed: int | None = DEFAULT_COMPUTE_SEED,
    models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a job config dict the CLI can run.

    Defaults the transformation list to both image algorithms —
    ``horizontal_flip`` and ``crop_resize`` at ``fraction: 0.5`` — each applied
    independently to the same selection, and lets callers override the loader
    set (e.g. the three column forms), the transformation list (sweep entries
    via `build_sweep`), the batch knobs, and the outputs shape.

    Args:
        loaders: Loader entry dicts (use `build_loader`). Required: one loader
            per selection is the intended shape, and the parquet path only
            exists in fixtures (`raw_bytes_data` / `path_data`).
        transformations: Transformation entry dicts ``{"type": ..., **params}``;
            defaults to the two image algorithms.
        transform_batch_size: ``operations.transform_batch_size`` knob.
        output_root: Directory under which ``output_path``/``samples_dir``
            default to ``<root>/ledger/{selection}.parquet`` and
            ``<root>/{selection}`` when not given explicitly.
        output_path: Ledger path pattern (``{selection}`` substitutable),
            overriding the ``output_root`` default.
        samples_dir: Payload-store pattern (``{selection}`` substitutable),
            overriding the ``output_root`` default.
        write_samples: Emit hashed payload files, or trace-only when ``False``.
        flush_batch_size: Output rows buffered before a write.
        include: Source columns passed through to the ledger (wildcards ok).
        exclude: Source columns withheld from the ledger (wins over include).
        compute_seed: ``compute.seed``; seeds sweep ``mode: random`` draws.
        models: Optional flat ``models:`` section (``name → {type, ...}`` ref),
            e.g. ``{"lama": {"type": "lama_inpaint"}}``.

    Returns:
        A nested config dict matching the ``JobConfig`` schema.

    Raises:
        ValueError: If neither ``output_root`` nor explicit ``output_path`` /
            ``samples_dir`` is given (the config needs somewhere to write).
    """
    if output_path is None or samples_dir is None:
        if output_root is None:
            raise ValueError(
                "build_config needs an output_root, or explicit output_path and samples_dir"
            )
        root = Path(output_root)
        if output_path is None:
            output_path = str(root / "ledger" / f"{SELECTION_NAME}.parquet")
        if samples_dir is None:
            samples_dir = str(root / SELECTION_NAME)

    outputs: dict[str, Any] = {
        "path": output_path,
        "samples_dir": samples_dir,
        "write_samples": write_samples,
        "flush_batch_size": flush_batch_size,
    }
    if include is not None:
        outputs["include"] = include
    if exclude is not None:
        outputs["exclude"] = exclude

    operations: dict[str, Any] = {
        "transformations": transformations
        or [
            {"type": "horizontal_flip"},
            {"type": "crop_resize", "fraction": 0.5},
        ],
        "outputs": outputs,
    }
    if transform_batch_size is not None:
        operations["transform_batch_size"] = transform_batch_size

    config: dict[str, Any] = {
        "compute": {"seed": compute_seed, "progress_bar": False},
        "dataloaders": {"loaders": loaders},
        "operations": operations,
    }
    if models is not None:
        config["models"] = models
    return config