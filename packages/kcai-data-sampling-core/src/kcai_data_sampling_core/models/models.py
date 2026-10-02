"""Named model references (NEW relative to dqm-ml).

The YAML ``models:`` section names reusable tool / target models; a
transformation references one by name (``target_model: yolo``) and the job
resolves the name to an instance via the models registry, so the YAML never
touches Python objects.
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelRefConfig(BaseModel):
    """A named reference to a model: a registered plugin, or the user's own code.

    The ``type`` is resolved against the loaded models registry at validation,
    so an unknown plugin name fails at config load — before the job ever
    touches weights. The reserved ``type: python`` is the bring-your-own-code
    route: it points the reference at the user's own file or module instead of
    a registered plugin (see the ``path``/``module``/``export`` fields), and
    *executes the referenced code by design* — the file builds its adapter.

    ``channels`` is the optional image-channel override for adapters that pin
    a channel count (like the inpainting tool, which declares RGB): an
    override that disagrees with the adapter's declared value is refused here,
    and the adapter's declared value stays the contract the instance enforces
    at run time.

    Attributes:
        type: A registered ``kcai_data_sampling.models`` plugin (a ``ToolModel``
            or ``TargetModel`` adapter), or the reserved value ``python``.
        path: ``type: python`` only — a local ``.py`` file with the adapter,
            resolved absolute first, else relative to the working directory.
        module: ``type: python`` only — an importable dotted module with the
            adapter (an alternative to ``path``, and the spelling for a
            package that needs relative imports).
        export: ``type: python`` only — dotted attribute path to the adapter
            class, factory callable, or instance; defaults to the ``models:``
            key name, then the sole model-shaped symbol in the source.
        weights: Optional weights/checkpoint the plugin may cache.
        params: Model-specific knobs.
        channels: Optional RGB/RGBA channel override (3 or 4). Defaults to the
            adapter's declared count when unset.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Registered model plugin name, or the reserved value ``python``.")
    path: str | None = Field(
        default=None,
        description="type: python only — local .py file, absolute first, else CWD-relative.",
    )
    module: str | None = Field(
        default=None,
        description="type: python only — importable dotted module with the adapter.",
    )
    export: str | None = Field(
        default=None,
        description=(
            "type: python only — dotted attribute path to export; defaults to the "
            "models: key name, then the sole model-shaped symbol."
        ),
    )
    weights: str | None = Field(default=None, description="Weights/checkpoint file name.")
    params: dict[str, Any] = Field(default_factory=dict, description="Model-specific knobs.")
    channels: Literal[3, 4] | None = Field(
        default=None,
        description="RGB/RGBA channel override; must agree with the adapter's declared count.",
    )

    @field_validator("type")
    @classmethod
    def _known(cls, v: str) -> str:
        """Refuse model types the registry does not know.

        ``python`` is the reserved bring-your-own-code kind and is the only
        value that skips the registry check (its own field rules run in
        ``_python_kind``).

        Args:
            v: The ``type`` string from the config.

        Returns:
            The validated type string.

        Raises:
            ValueError: If the type is neither ``python`` nor registered.
        """
        if v == "python":
            return v
        from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry

        registry = PluginLoadedRegistry.get_models_registry()
        if v not in registry:
            known = ", ".join(sorted(registry)) or "none"
            raise ValueError(f"unknown model type {v!r} (registered models: {known})")
        return v

    @model_validator(mode="after")
    def _python_kind(self) -> Self:
        """Enforce the reserved ``python`` kind's field rules.

        A ``type: python`` reference needs exactly one of ``path``/``module``
        and may carry ``export`` besides the ordinary ``weights``/``params``/
        ``channels``. Every other type is a registered plugin, for which the
        source fields are meaningless and refused up front.

        Returns:
            The validated model reference.

        Raises:
            ValueError: If a ``python`` reference misses or duplicates
                ``path``/``module``, or a plugin reference carries a source
                field.
        """
        if self.type != "python":
            for field in ("path", "module", "export"):
                if getattr(self, field) is not None:
                    raise ValueError(
                        f"model type {self.type!r} is a registered plugin; '{field}' is only valid for type: python"
                    )
            return self
        if (self.path is None) == (self.module is None):
            raise ValueError(
                "type: python requires exactly one of 'path' (a .py file) or 'module' (an importable dotted path)"
            )
        return self

    @model_validator(mode="after")
    def _channels(self) -> Self:
        """Refuse a channel override that disagrees with the adapter's count.

        Only adapters that declare a pinned ``channels`` attribute are
        opinionated; an override that matches the declared count is the
        explicit version of the default and is accepted.

        Returns:
            The validated model reference.

        Raises:
            ValueError: If ``channels`` conflicts with the adapter's declared
                count.
        """
        if self.channels is None:
            return self
        from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry

        declared = getattr(PluginLoadedRegistry.get_models_registry().get(self.type), "channels", None)
        if declared is not None and self.channels != declared:
            raise ValueError(
                f"model {self.type!r} declares channels={declared!r}; "
                f"the config overrides channels={self.channels!r} in conflict"
            )
        return self


class ModelsConfig(BaseModel):
    """Named collection of model references.

    The ``models:`` section of a job is written flat — ``models: {lama:
    {type: ...}}`` — and re-shaped onto the ``models`` field below; the
    doubly-nested form (``models.models``) is accepted too.

    Attributes:
        models: Mapping from model name to a model reference; transformations
            reference models by these names.
    """

    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelRefConfig] = Field(description="Named model references.")

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_section(cls, data: Any) -> Any:
        """Re-shape a flat ``models:`` section onto the ``models`` field.

        The YAML section names its models directly; without this, the single
        field would demand a doubly-nested ``models: {models: {name: ref}}``.
        The nested form keeps working.

        Args:
            data: The raw section dict (or already-nested shape).

        Returns:
            ``{"models": data}`` when the section is the flat name→reference
            mapping, else ``data`` unchanged.
        """
        if isinstance(data, dict) and "models" not in data:
            return {"models": data}
        return data

    @model_validator(mode="after")
    def _resolve_python_references(self) -> Self:
        """Refuse, at load, a ``type: python`` reference whose source cannot resolve.

        A ``type: python`` entry names the user's own code: the file/module is
        loaded and its export resolved here, so a missing file, an empty
        ``export``, or an unresolvable default fail the config load instead of
        surfacing mid-run. Loading executes the referenced code by design (the
        bring-your-own-code feature); the adapter's real role conformance is
        still enforced at construction, against the role the transformation
        declares.

        Returns:
            The validated models section.

        Raises:
            ValueError: If a ``type: python`` reference's source cannot be
                loaded or its export resolved.
        """
        if not self.models:
            return self
        from kcai_data_sampling_core.utils.registry import load_model_source

        for key, ref in self.models.items():
            if ref.type != "python":
                continue
            load_model_source(key, ref.path, ref.module, ref.export)
        return self
