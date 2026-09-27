"""Plugin registry: discover transformations, models, loaders and writers.

Algorithms and IO plugins are discovered from Python entry points
(``kcai_data_sampling.transformations``, ``kcai_data_sampling.models``,
``kcai_data_sampling.dataloaders``, ``kcai_data_sampling.outputwriter``) and
lazily cached by :class:`PluginLoadedRegistry`. New packages plug in without
touching core: each algorithm carries its config schema next to it, and plugins
resolve themselves through the registry.
"""

import hashlib
import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from kcai_data_sampling_core.api.transformation import Transformation

logger = logging.getLogger(__name__)

_MODEL_PROTOCOL_METHODS = ("grad", "inpaint")


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


def _model_shaped(obj: Any) -> bool:
    """Probe whether a symbol is shaped like a model adapter.

    The union of the role protocols' *shape* — a non-empty ``name`` plus at
    least one of the model methods (``grad`` for the target role, ``inpaint``
    for the tool role) — so the probe works on a class or an instance without
    committing to a role. Real conformance is still enforced by
    ``check_model`` at construction, against the role the transformation
    declares.

    Args:
        obj: A module attribute to probe.

    Returns:
        True if the symbol exposes a non-empty string ``name`` and at least one
        of the model protocol methods.
    """
    if inspect.ismodule(obj):
        return False
    name = getattr(obj, "name", None)
    if not isinstance(name, str) or not name:
        return False
    return any(callable(getattr(obj, method, None)) for method in _MODEL_PROTOCOL_METHODS)


def _model_candidate_symbols(module: ModuleType) -> dict[str, Any]:
    """Map the module's public, model-shaped attributes.

    Args:
        module: The loaded user source module.

    Returns:
        The public (non-underscore) attributes that pass :func:`_model_shaped`,
        keyed by attribute name.
    """
    return {
        symbol: obj
        for symbol, obj in vars(module).items()
        if not symbol.startswith("_") and _model_shaped(obj)
    }


def _resolve_export(module: ModuleType, name: str, export: str | None, origin: str) -> Any:
    """Resolve the symbol a ``type: python`` reference exports.

    An explicit ``export`` is a dotted attribute path into the module and wins
    outright. Unset, the module attribute named like the ``models:`` key is
    used; else the sole symbol shaped like a model adapter; else the load
    fails loudly, listing the candidates, so the user knows what to pin.

    Args:
        module: The loaded user source module.
        name: The ``models:`` entry name (candidate attribute name).
        export: The dotted attribute path to export, or None for the default
            resolution.
        origin: The configured source (file path or dotted module) the user
            wrote, for the error messages.

    Returns:
        The exported symbol: a class, a factory callable, or an instance.

    Raises:
        ValueError: If an ``export`` names nothing, the default resolution is
            ambiguous, or no model symbol is exposed at all.
    """
    if export is not None:
        obj: Any = module
        for part in export.split("."):
            if not hasattr(obj, part):
                raise ValueError(
                    f"model {name!r}: export {export!r} not found in {origin}"
                )
            obj = getattr(obj, part)
        return obj
    if hasattr(module, name):
        return getattr(module, name)
    candidates = _model_candidate_symbols(module)
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    if candidates:
        raise ValueError(
            f"model {name!r}: {origin} exposes several model candidates "
            f"({', '.join(sorted(candidates))}); set 'export:' to pick the adapter"
        )
    public = sorted(
        s for s in vars(module) if not s.startswith("_") and not inspect.ismodule(getattr(module, s))
    )
    raise ValueError(
        f"model {name!r}: {origin} exposes no model candidate to default to; "
        f"set 'export:' to one of the module's symbols ({', '.join(public) or 'none'})"
    )


def _load_path_module(name: str, path: str) -> ModuleType:
    """Load a local ``.py`` file as a module, absolute-first, else CWD-relative.

    The module name is derived from the resolved absolute path, so the same
    file is loaded once and later calls reuse it (``sys.modules``) without
    re-executing the file.

    Args:
        name: The ``models:`` entry name, for the error messages.
        path: The configured file path (absolute, or relative to the working
            directory).

    Returns:
        The loaded module.

    Raises:
        ValueError: If the resolved path is not a readable file, or executing
            it raises.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    if not resolved.is_file():
        raise ValueError(f"model {name!r}: type: python file not found: {resolved}")
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    modname = f"_kcai_user_model_{digest}"
    cached = sys.modules.get(modname)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(modname, resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"model {name!r}: cannot load {resolved} as a python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        sys.modules.pop(modname, None)
        raise ValueError(f"model {name!r}: {resolved} raised while loading: {exc}") from exc
    return module


def load_model_source(
    name: str,
    path: str | None = None,
    module: str | None = None,
    export: str | None = None,
) -> Any:
    """Load a user model source and return the exported symbol.

    The bring-your-own-code route: a ``models:`` entry with ``type: python``
    points at the user's own code instead of a registered plugin. Loading the
    source **executes it** — that is the feature, and the intent, of a
    ``type: python`` reference: the referenced file builds its adapter. The
    file is user-supplied and named from the user's own config; nothing is
    downloaded.

    ``path`` is a local ``.py`` file, resolved absolute first, else relative
    to the working directory; a single file cannot use relative imports
    (prefer ``module`` for a package). ``module`` is an importable dotted
    path. Exactly one of the two is required.

    exported symbol may be a class, a factory callable, or an instance;
    see :func:`_resolve_export` for how ``export`` defaults.

    Args:
        name: The ``models:`` entry name.
        path: A local ``.py`` file holding the adapter.
        module: An importable dotted module holding the adapter.
        export: Dotted attribute path into the source; default resolution per
            :func:`_resolve_export`.

    Returns:
        The exported symbol (class, factory callable, or instance).

    Raises:
        ValueError: If neither **nor both** of ``path``/``module`` are given,
            the source cannot be loaded, or the export resolution is
            ambiguous or empty.
    """
    if (path is None) == (module is None):
        raise ValueError(f"model {name!r}: type: python needs exactly one of 'path' or 'module'")
    if path is not None:
        loaded = _load_path_module(name, path)
        origin = str(Path(path).resolve())
    else:
        origin = str(module)
        try:
            loaded = importlib.import_module(module)
        except ImportError as exc:
            raise ValueError(f"model {name!r}: cannot import module {module!r}: {exc}") from exc
    return _resolve_export(loaded, name, export, origin)


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


def register_model(name: str, model: Any) -> None:
    """Register a model adapter in-process, visible to the ``models:`` section.

    Lets a config name an adapter already loaded in the kernel — notebooks,
    the gate smoke, tests — without shipping a plugin package. The
    registration seeds the loaded models registry, so a later
    ``models: {name: {type: <name>}}`` entry resolves through the ordinary
    ``type``-validation and ``_build_models`` path. It is purely additive:
    nothing is persisted.

    An entry-point plugin owns its name and is never masked: the plugin
    registry is (re)loaded before the registration, and a name the plugin
    already claims is refused loudly rather than silently shadowed.

    Args:
        name: The ``type`` value the ``models:`` section will use.
        model: The adapter (a class, factory callable, or instance).

    Raises:
        ValueError: If ``name`` is already claimed by an entry-point plugin.
    """
    registry = PluginLoadedRegistry.get_models_registry()
    if name in registry:
        raise ValueError(
            f"model name {name!r} is already claimed by an entry-point plugin; "
            f"register a different name and update the models: section"
        )
    registry[name] = model


def get_transformations_registry() -> dict[str, type[Transformation]]:
    """Convenience alias for :meth:`PluginLoadedRegistry.get_transformations_registry`."""
    return PluginLoadedRegistry.get_transformations_registry()