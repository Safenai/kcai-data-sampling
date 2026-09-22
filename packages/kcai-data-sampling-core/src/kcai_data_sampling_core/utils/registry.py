"""Plugin registry: discover transformations, models, loaders and writers.

Algorithms and IO plugins are discovered from Python entry points
(``kcai_data_sampling.transformations``, ``kcai_data_sampling.models``,
``kcai_data_sampling.dataloaders``, ``kcai_data_sampling.outputwriter``) and
lazily cached by :class:`PluginLoadedRegistry`. New packages plug in without
touching core: each algorithm carries its config schema next to it, and plugins
resolve themselves through the registry.
"""

import logging
import sys
from typing import Any

from kcai_data_sampling_core.api.transformation import Transformation

logger = logging.getLogger(__name__)


def load_registered_plugins(plugin_group: str, base_class: Any, base_name: str = "default") -> dict[str, Any]:
    """Discover and load plugins registered via Python entry points.

    Args:
        plugin_group: The entry point group name (e.g.
            ``kcai_data_sampling.transformations``).
        base_class: Optional base class to verify plugin type safety.
        base_name: Name of the base-class entry to ignore during discovery.

    Returns:
        A dictionary mapping plugin names to their loaded classes.
    """
    try:
        # python 3.10+
        plugin_entry_points = __import__("importlib.metadata").metadata.entry_points(group=plugin_group)
    except TypeError:
        # Old python version not supported
        logger.warning("Old python version not supported: %s", sys.version_info)
        return {}

    registry: dict[str, Any] = {}
    for v in plugin_entry_points:
        # Filter base class registry (not callable)
        if v.name != base_name:
            obj = v.load()
            if base_class is None or issubclass(obj, base_class):
                logger.debug("Referencing %s - %s class %s from %s", plugin_group, v.name, obj, base_class)
                registry[v.name] = obj
            else:
                logger.error(
                    "Entry point %s - %s class %s not derived from %s ignored",
                    plugin_group,
                    v.name,
                    obj,
                    base_class,
                )
    return registry


class PluginLoadedRegistry:
    """Singleton registry that provides lazy access to all registered plugins.

    Each registry is loaded once from its entry-point group and cached; the
    ``get_*_registry`` class methods trigger the load.
    """

    _transformations_registry: dict[str, type[Transformation]] | None = None
    _models_registry: dict[str, Any] | None = None
    _dataloaders_registry: dict[str, Any] | None = None
    _outputwriter_registry: dict[str, Any] | None = None

    @classmethod
    def get_transformations_registry(cls) -> dict[str, type[Transformation]]:
        """Return the registry of available transformations.

        Returns:
            A dictionary mapping transformation names to algorithm classes.
        """
        if not cls._transformations_registry:
            cls._transformations_registry = load_registered_plugins(
                "kcai_data_sampling.transformations", Transformation
            )
        return cls._transformations_registry

    @classmethod
    def get_models_registry(cls) -> dict[str, Any]:
        """Return the registry of available model adapters.

        Returns:
            A dictionary mapping model type names to adapter classes.
        """
        if not cls._models_registry:
            cls._models_registry = load_registered_plugins("kcai_data_sampling.models", None)
        return cls._models_registry

    @classmethod
    def get_dataloaders_registry(cls) -> dict[str, Any]:
        """Return the registry of available dataloaders.

        Returns:
            A dictionary mapping loader type names to loader classes.
        """
        if not cls._dataloaders_registry:
            cls._dataloaders_registry = load_registered_plugins("kcai_data_sampling.dataloaders", None)
        return cls._dataloaders_registry

    @classmethod
    def get_outputwriter_registry(cls) -> dict[str, Any]:
        """Return the registry of available output writers.

        Returns:
            A dictionary mapping writer type names to writer classes.
        """
        if not cls._outputwriter_registry:
            cls._outputwriter_registry = load_registered_plugins("kcai_data_sampling.outputwriter", None)
        return cls._outputwriter_registry


def get_transformations_registry() -> dict[str, type[Transformation]]:
    """Convenience alias for :meth:`PluginLoadedRegistry.get_transformations_registry`."""
    return PluginLoadedRegistry.get_transformations_registry()