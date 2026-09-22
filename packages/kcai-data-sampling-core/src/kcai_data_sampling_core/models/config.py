"""Root job configuration, and the registry-resolved transformation config.

The ``transformations`` entries of the YAML are validated against each
algorithm's own registered pydantic schema: core
cannot enumerate the discriminated union because algorithm packages arrive
after core, so the ``type`` string is looked up in the loaded transformation
registry and the entry is validated against the registered schema. Unknown or
mistyped types are refused at config load.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kcai_data_sampling_core.models.dataloaders import DataLoadersConfig
from kcai_data_sampling_core.models.global_ import ComputeConfig, ErrorsConfig, StorageConfig
from kcai_data_sampling_core.models.interfaces import SamplingInterfaceConfig
from kcai_data_sampling_core.models.models import ModelsConfig


class TransformationConfig(BaseModel):
    """Base transformation configuration; algorithm packages subclass it.

    The base carries the config keys shared by every transformation and
    validates ``type`` against the loaded registry. Each algorithm package
    subclasses this with its own fields (and pins ``type`` to its literal),
    and registers the **algorithm class** that carries the subclass as its
    ``Config`` attribute — that is how config validation finds the specific
    schema.

    Attributes:
        name: Unique name of the transformation within the interface.
        type: Registry-resolved algorithm type (the transformation's identity).
        seed: Random seed; always accepted but only drawn when the algorithm
            is stochastic.
        storage: Optional storage override.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique name within the interface.")
    type: str = Field(description="Registry-resolved transformation type.")
    seed: int | None = Field(default=None, description="Random seed; drawn only when stochastic.")
    storage: bool | dict[str, Any] | None = Field(
        default=None,
        description="Optional storage override (local only this phase).",
    )

    @field_validator("type")
    @classmethod
    def _known(cls, v: str) -> str:
        """Refuse transformation types the registry does not know.

        Args:
            v: The ``type`` string from the config.

        Returns:
            The validated type string.

        Raises:
            ValueError: If the type is not registered.
        """
        from kcai_data_sampling_core.utils.registry import get_transformations_registry

        if v not in get_transformations_registry():
            raise ValueError(f"unknown transformation type {v!r}")
        return v


class JobConfig(BaseModel):
    """Root configuration for a sample-generation job. Each field maps to a pipeline stage.

    Attributes:
        storage: Global storage config (local only this phase).
        compute: Global compute / runtime settings.
        errors: Global error-handling policy.
        dataloaders: The data loaders and their selections.
        models: Named tool / target model references.
        transformations: The single interface: ordered transformations,
            outputs, and the batch/error/storage overrides.
    """

    model_config = ConfigDict(extra="forbid")

    storage: StorageConfig | None = None
    compute: ComputeConfig | None = None
    errors: ErrorsConfig | None = None
    dataloaders: DataLoadersConfig = Field(description="Data loaders.")
    models: ModelsConfig | None = None
    transformations: SamplingInterfaceConfig = Field(description="The single transformations interface.")

    @field_validator("transformations")
    @classmethod
    def _resolve_transformations(cls, v: SamplingInterfaceConfig) -> SamplingInterfaceConfig:
        """Resolve each raw transformation entry against its registered schema.

        The YAML entries are plain dicts; this validator looks each ``type`` up
        in the transformation registry, takes the algorithm's own config schema
        (``algorithm.Config``), and validates the entry against it, so required
        parameters and unknown fields are refused at config load.

        Args:
            v: The partially validated interface config.

        Returns:
            The interface config with every transformation validated against
            its specific registered schema.

        Raises:
            ValueError: If a transformation type is unknown or an entry does
                not pass its algorithm's schema.
        """
        from kcai_data_sampling_core.utils.registry import get_transformations_registry

        registry = get_transformations_registry()
        resolved: list[TransformationConfig] = []
        for entry in v.transformations:
            entry = dict(entry)
            algo = registry.get(entry.get("type", ""))
            if algo is None:
                raise ValueError(f"unknown transformation type {entry.get('type')!r}")
            config_cls = getattr(algo, "Config", TransformationConfig)
            resolved.append(config_cls.model_validate(entry))
        v.transformations = resolved
        return v