"""Command-line interface for running jobs from YAML configuration files.

Port of dqm-ml's ``cli.py`` to the single-transformations-interface shape:
one interface with an ordered list of transformations, an outputs
section (ledger + payload store), and error/batch overrides.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_core.utils.registry import (
    PluginLoadedRegistry,
    get_transformations_registry,
)
from kcai_data_sampling_job.job import SamplingJob
from kcai_data_sampling_job.utils.shared import merge_errors
import yaml

logger = logging.getLogger(__name__)


def parse_args(arg_list: list[str] | None) -> Any:
    """Parse the job CLI arguments.

    Args:
        arg_list: List of arguments (default: ``sys.argv[1:]``).

    Returns:
        The parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        prog="kcai-data-sampling-job",
        description="kcai data-sampling job client",
        epilog="Run a sample-generation job from a YAML configuration.",
    )
    parser.add_argument(
        "-p",
        "--process-config",
        type=str,
        nargs="+",
        required=True,
        help="configuration files to execute",
    )
    parser.add_argument(
        "--save-config",
        type=str,
        help="path to save the resolved configuration",
    )
    return parser.parse_args(arg_list)


def execute(arg_list: list[str] | None = None) -> None:
    """Run a job from one or more YAML configuration files.

    Args:
        arg_list: Command-line arguments (default: ``sys.argv[1:]``).
    """
    args = parse_args(arg_list)
    config: dict[str, Any] = {}

    for config_file in args.process_config:
        config_path = Path(config_file).resolve()
        if not config_path.is_file():
            logger.error("Config file does not exist: %s", config_file)
            return
        with config_path.open() as stream:
            try:
                config.update(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                logger.error("Failed to parse job configuration: %s", config_file)
                print(exc)
                return

    if args.save_config:
        save_path = Path(args.save_config).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as stream:
            yaml.safe_dump(config, stream)

    run(config)


def _resolve_compute_config(validated: JobConfig) -> dict[str, Any]:
    """Resolve the compute settings, providing defaults when not specified.

    Args:
        validated: The validated job configuration.

    Returns:
        The compute settings as a dict.
    """
    compute = validated.compute
    return {
        "seed": compute.seed if compute else 42,
        "log_level": compute.log_level if compute else "warning",
        "max_memory": compute.max_memory if compute else None,
        "device": compute.device if compute else "auto",
        "progress_bar": compute.progress_bar if compute else True,
        "threads": compute.threads if compute else 4,
    }


def _build_images_error_dict(validated: JobConfig) -> dict[str, str] | None:
    """Extract the image error policy used by the loaders.

    Args:
        validated: The validated job configuration.

    Returns:
        A dict with the two image error keys, or ``None`` for the loader
        defaults.
    """
    merged = merge_errors(validated.errors, validated.transformations.errors)
    if merged.images is None:
        return None
    images = merged.images
    return {
        "on_decode_failure": images.on_decode_failure,
        "on_unsupported_format": images.on_unsupported_format,
    }


def _build_writers(
    validated: JobConfig,
    outputs_registry: dict[str, Any],
) -> tuple[Any | None, Any | None]:
    """Build the payload and ledger writers from the outputs config.

    Args:
        validated: The validated job configuration.
        outputs_registry: The registered ``kcai_data_sampling.outputwriter``
            plugins.

    Returns:
        A ``(payload_writer, ledger_writer)`` tuple; each may be ``None``.
    """
    outputs = validated.transformations.outputs

    ledger_writer = None
    if "parquet" in outputs_registry:
        ledger_writer = outputs_registry["parquet"](
            name="ledger",
            config={"path_pattern": outputs.path, "flush_batch_size": outputs.flush_batch_size},
        )
    else:
        logger.warning("Output writer type 'parquet' not found in the registry; no ledger is written.")

    payload_writer = None
    if "images" in outputs_registry:
        payload_writer = outputs_registry["images"](
            name="images",
            config={"images_dir": outputs.images_dir, "write_images": outputs.write_images},
        )
    else:
        logger.warning("Output writer type 'images' not found in the registry; trace-only run.")

    return payload_writer, ledger_writer


def _build_transformations(
    validated: JobConfig,
    transformations_registry: dict[str, Any],
) -> list[Any]:
    """Instantiate the ordered transformations for the interface.

    Each validated entry is its algorithm's own config instance; the extra
    config keys (``name``, ``type``, ``seed``, ``storage``) are split off and
    only the resolved algorithm parameters reach the transformation.

    Args:
        validated: The validated job configuration.
        transformations_registry: The registered transformations.

    Returns:
        An ordered list of transformation instances.
    """
    instances: list[Any] = []
    for entry in validated.transformations.transformations:
        algorithm = transformations_registry[entry.type]
        dumped = entry.model_dump()
        params = {k: v for k, v in dumped.items() if k not in ("name", "type", "seed", "storage")}
        instances.append(algorithm(config={"seed": dumped.get("seed"), **params}))
    return instances


def run(config: dict[str, Any]) -> dict[str, int]:
    """Execute a job from a validated configuration dictionary.

    Args:
        config: The job configuration.

    Returns:
        The per-selection output row counts.

    Raises:
        ValueError: If the configuration is empty or invalid.
    """
    if not config:
        raise ValueError("Job requires a configuration dictionary.")

    validated = JobConfig.model_validate(config)

    dataloaders_registry = PluginLoadedRegistry.get_dataloaders_registry()
    outputs_registry = PluginLoadedRegistry.get_outputwriter_registry()
    transformations_registry = get_transformations_registry()

    compute = _resolve_compute_config(validated)
    if compute["log_level"]:
        logging.basicConfig(level=getattr(logging, compute["log_level"].upper()))

    images_errors = _build_images_error_dict(validated)

    dataloaders: dict[str, Any] = {}
    for loader_cfg in validated.dataloaders.loaders:
        loader_cls = dataloaders_registry.get(loader_cfg.type)
        if loader_cls is None:
            raise ValueError(f"unknown dataloader type {loader_cfg.type!r}")
        dataloaders[loader_cfg.name] = loader_cls(
            name=loader_cfg.name,
            config=loader_cfg,
            errors=images_errors,
            threads=compute["threads"],
        )

    payload_writer, ledger_writer = _build_writers(validated, outputs_registry)
    transformations = _build_transformations(validated, transformations_registry)

    job = SamplingJob(
        dataloaders=dataloaders,
        transformations=transformations,
        payload_writer=payload_writer,
        ledger_writer=ledger_writer,
        errors=merge_errors(validated.errors, validated.transformations.errors),
        progress_bar=compute["progress_bar"],
        transform_batch_size=validated.transformations.transform_batch_size,
    )

    return job.run()


if __name__ == "__main__":
    execute()