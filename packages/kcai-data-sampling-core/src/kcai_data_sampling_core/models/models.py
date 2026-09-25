"""Named model references (NEW relative to dqm-ml).

The YAML ``models:`` section names reusable tool / target models; a
transformation references one by name (``target_model: yolo``) and the job
resolves the name to an instance via the models registry, so the YAML never
touches Python objects.
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelRefConfig(BaseModel):
    """A named reference to a registered model plugin.

    The ``type`` is resolved against the loaded models registry at
    validation, so an unknown plugin name fails at config load — before the
    job ever touches weights. ``channels`` is the optional image-channel
    override for adapters that pin a channel count (like the inpainting tool,
    which declares RGB): an override that disagrees with the adapter's
    declared value is refused here, and the adapter's declared value stays the
    contract the instance enforces at run time.

    Attributes:
        type: Registered ``kcai_data_sampling.models`` plugin (a ``ToolModel``
            or ``TargetModel`` adapter).
        weights: Optional weights/checkpoint the plugin may cache.
        params: Plugin-specific knobs.
        channels: Optional RGB/RGBA channel override (3 or 4). Defaults to the
            adapter's declared count when unset.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Registered model plugin name.")
    weights: str | None = Field(default=None, description="Weights/checkpoint file name.")
    params: dict[str, Any] = Field(default_factory=dict, description="Plugin-specific knobs.")
    channels: Literal[3, 4] | None = Field(
        default=None,
        description="RGB/RGBA channel override; must agree with the adapter's declared count.",
    )

    @field_validator("type")
    @classmethod
    def _known(cls, v: str) -> str:
        """Refuse model types the registry does not know.

        Args:
            v: The ``type`` string from the config.

        Returns:
            The validated type string.

        Raises:
            ValueError: If the type is not registered.
        """
        from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry

        registry = PluginLoadedRegistry.get_models_registry()
        if v not in registry:
            known = ", ".join(sorted(registry)) or "none"
            raise ValueError(f"unknown model type {v!r} (registered models: {known})")
        return v

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