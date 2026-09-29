"""Dependency management: versions, optional dependencies, command discovery."""

from collections.abc import Generator
from contextlib import contextmanager
import logging
from typing import Any

from kcai_data_sampling_core._version_ import __version__ as core_version
from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry

logger = logging.getLogger(__name__)


@contextmanager
def optional_dependencies(error: str = "ignore") -> Generator[None, None, None]:
    """Context manager for handling optional dependencies.

    Args:
        error: How to handle a missing optional dependency: ``"ignore"`` keeps
            going (default), ``"warn"`` prints a warning, ``"raise"`` re-raises
            the ``ImportError``.

    Yields:
        None; this context manager offers no value.
    """
    assert error in {"raise", "warn", "ignore"}
    try:
        yield None
    except ImportError as exc:
        if error == "raise":
            raise exc
        if error == "warn":
            print(f"Warning: missing optional dependency {exc.name}. Use pip to install.")


def _job_version() -> str | None:
    """Return the installed job package version, or ``None`` if absent."""
    try:
        from kcai_data_sampling_job._version_ import __version__

        return __version__
    except ImportError:
        return None


def _images_version() -> str | None:
    """Return the installed images package version, or ``None`` if absent."""
    try:
        from kcai_data_sampling_images._version_ import __version__

        return __version__
    except ImportError:
        return None


def display_version(arg_list: list[str] | None = None) -> None:
    """Print the installed package versions to stdout.

    Args:
        arg_list: Unused, provided for CLI compatibility.
    """
    del arg_list
    versions = {
        "kcai-data-sampling": __import__("kcai_data_sampling").__version__,
        "core": core_version,
        "job": _job_version(),
        "images": _images_version(),
    }
    for name, version in versions.items():
        print(f"{name}: {version}")


def display_list_of(arg_list: list[str] | None = None) -> None:
    """Print all registered plugins (transformations, loaders, writers).

    Args:
        arg_list: Unused, provided for CLI compatibility.
    """
    del arg_list
    groups = (
        ("transformations", PluginLoadedRegistry.get_transformations_registry()),
        ("models", PluginLoadedRegistry.get_models_registry()),
        ("dataloaders", PluginLoadedRegistry.get_dataloaders_registry()),
        ("output_writers", PluginLoadedRegistry.get_outputwriter_registry()),
    )
    for group_name, registry in groups:
        print(f"Available data {group_name}")
        for key, value in registry.items():
            print(f"- {key} - {value.__module__}.{value.__name__}")


def get_available_command() -> dict[str, Any]:
    """Build the dictionary of available CLI commands.

    Returns:
        A dict mapping command names to handler functions.
    """
    command_list: dict[str, Any] = {"version": display_version, "list": display_list_of}

    with optional_dependencies("warn"):
        from kcai_data_sampling_job.cli import execute

        command_list["process"] = execute

    return command_list
