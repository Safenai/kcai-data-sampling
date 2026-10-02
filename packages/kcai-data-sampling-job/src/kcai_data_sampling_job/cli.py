"""Command-line interface for running jobs from YAML configuration files.

Port of dqm-ml's ``cli.py`` to the single-transformations-interface shape:
one interface with a list of transformations each applied independently to the
same selection, an outputs section (ledger + payload store), and
error/batch overrides.
"""

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry, get_transformations_registry, load_model_source
import yaml

from kcai_data_sampling_job.job import SamplingJob
from kcai_data_sampling_job.utils.shared import merge_errors

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
    merged = merge_errors(validated.errors, validated.operations.errors)
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
    outputs = validated.operations.outputs

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
            config={
                "samples_dir": outputs.samples_dir,
                "write_samples": outputs.write_samples,
                "flush_batch_size": outputs.flush_batch_size,
            },
        )
    else:
        logger.warning("Output writer type 'images' not found in the registry; trace-only run.")

    return payload_writer, ledger_writer


def _available_models(models: dict[str, Any]) -> str:
    """The declared ``models:`` names, sorted, or ``'none declared'`` when empty.

    Args:
        models: The built model instances, keyed by their ``models:`` names.

    Returns:
        The comma-joined names, or ``'none declared'``.
    """
    return ", ".join(sorted(models)) or "none declared"


def _build_model(name: str, ref: Any, registry: dict[str, Any]) -> Any:
    """Instantiate one ``models:`` entry into an adapter instance.

    ``weights=ref.weights`` is dropped when unset; the plugin's own knobs ride
    along as ``**ref.params``. A ``type: python`` entry sources the user's own
    code instead (its file/module and export were already resolved at config
    load): the exported class or factory function is constructed with the same
    conventions, and an exported instance is used as-is.

    Args:
        name: The ``models:`` entry name.
        ref: The validated model reference.
        registry: The loaded models registry (plugin adapters).

    Returns:
        The adapter instance.

    Raises:
        ValueError: If a model ``type`` is not registered (belt-and-braces:
            config load already refuses unknown types).
    """
    kwargs = {} if ref.weights is None else {"weights": ref.weights}
    if ref.type == "python":
        export = load_model_source(name, ref.path, ref.module, ref.export)
        if inspect.isclass(export) or inspect.isfunction(export):
            return export(**kwargs, **ref.params)
        return export
    adapter = registry.get(ref.type)
    if adapter is None:
        raise ValueError(
            f"model {name!r}: unknown type {ref.type!r} (registered models: {', '.join(sorted(registry)) or 'none'})"
        )
    return adapter(**kwargs, **ref.params)


def _build_models(validated: JobConfig) -> dict[str, Any]:
    """Instantiate the named ``models:`` references into a name→adapter map.

    Each reference is ``{type, weights, params, channels}``: the adapter class
    comes from the loaded models registry and is constructed with
    ``weights=ref.weights`` (dropped when unset) plus ``**ref.params`` — the
    plugin's own knobs. The optional ``channels`` override is enforced at
    config load (``ModelRefConfig`` refuses a conflict with the adapter's
    declared count) and by the adapter at run time against the actual batch.

    A ``type: python`` reference names the user's own source instead (its
    file/module and export were already resolved at config load); see
    :func:`_build_model`.

    Args:
        validated: The validated job configuration.

    Returns:
        A mapping of model names (as referenced by transformations) to adapter
        instances; empty when no ``models:`` section is present.
    """
    if validated.models is None:
        return {}
    registry = PluginLoadedRegistry.get_models_registry()
    return {name: _build_model(name, ref, registry) for name, ref in validated.models.models.items()}


_SLOT_BY_ROLE = {"tool": "tool_model", "target": "target_model"}

#: Config keys split off before the resolved algorithm parameters reach the
#: transformation; they are the base schema's, not the algorithm's.
_EXCLUDED_PARAMS = ("name", "type", "seed", "storage", "columns")


def _bind_role_model(entry: Any, role: str | None, models: dict[str, Any]) -> dict[str, Any]:
    """Resolve a transformation's model role to its adapter instance.

    A model-bearing algorithm (``tool``/``target`` role) references its model
    by name in the config; the name is checked against the built ``models``
    map and the adapter instance lands under the base class's slot key, so the
    base class's slot/exactly-one checks run as usual.

    Args:
        entry: The validated transformation entry.
        role: The algorithm's model role (``tool``/``target``), or ``None``.
        models: The built model instances, keyed by their ``models:`` names.

    Returns:
        The ``{slot: adapter}`` params to inject, or ``{}`` when the algorithm
        carries no model role.

    Raises:
        ValueError: If the algorithm names an unknown model, or omits the
            model name outright.
    """
    if role is None:
        return {}
    slot = _SLOT_BY_ROLE[role]
    name = getattr(entry, slot, None)
    if name is None:
        raise ValueError(
            f"{entry.type!r} needs a model ({role} role): set '{slot}:' in the "
            f"transformation config to a name from the models: section "
            f"({_available_models(models)})"
        )
    model = models.get(name)
    if model is None:
        raise ValueError(
            f"{entry.type!r}: unknown {slot} model {name!r} (models: section names: {_available_models(models)})"
        )
    return {slot: model}


def _build_transformations(
    validated: JobConfig,
    transformations_registry: dict[str, Any],
    models: dict[str, Any],
) -> list[Any]:
    """Instantiate the transformations for the interface.

    Each validated entry is its algorithm's own config instance; the extra
    config keys (``name``, ``type``, ``seed``, ``storage``, ``columns``) are
    split off and only the resolved algorithm parameters reach the
    transformation. An algorithm with a model role (``tool``/``target``)
    references its model by name; the name is resolved against the built
    ``models`` map (see :func:`_bind_role_model`).

    Args:
        validated: The validated job configuration.
        transformations_registry: The registered transformations.
        models: The built model instances, keyed by their ``models:`` names.

    Returns:
        A list of transformation instances, one per configured entry.
    """
    instances: list[Any] = []
    for raw_entry in validated.operations.transformations:
        entry = raw_entry
        algorithm = transformations_registry[entry.type]
        dumped = entry.model_dump()
        params = {k: v for k, v in dumped.items() if k not in _EXCLUDED_PARAMS}
        params.update(_bind_role_model(entry, getattr(algorithm, "model_role", None), models))
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
    for raw_loader_cfg in validated.dataloaders.loaders:
        loader_cfg = raw_loader_cfg
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
    models = _build_models(validated)
    transformations = _build_transformations(validated, transformations_registry, models)

    job = SamplingJob(
        dataloaders=dataloaders,
        transformations=transformations,
        payload_writer=payload_writer,
        ledger_writer=ledger_writer,
        errors=merge_errors(validated.errors, validated.operations.errors),
        progress_bar=compute["progress_bar"],
        transform_batch_size=validated.operations.transform_batch_size,
        include_columns=validated.operations.outputs.include,
        exclude_columns=validated.operations.outputs.exclude,
    )

    return job.run()


if __name__ == "__main__":
    execute()
