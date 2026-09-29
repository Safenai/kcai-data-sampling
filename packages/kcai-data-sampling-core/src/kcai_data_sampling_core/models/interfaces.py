"""The transformations interface configuration (NEW relative to dqm-ml).

There is exactly one pipeline stage — transform samples — so dqm-ml's three
interfaces collapse into one ``transformations`` interface.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kcai_data_sampling_core.models.global_ import ErrorsConfig, StorageConfig
from kcai_data_sampling_core.models.outputs import SamplingOutputsConfig
from kcai_data_sampling_core.models.transformation import TransformationConfig


class SamplingInterfaceConfig(BaseModel):
    """Configuration of the single ``transformations`` interface.

    Attributes:
        storage: Storage override for this interface.
        errors: Error-handling override for this interface.
        transform_batch_size: Target ``(B, *sample)`` batch handed to ``apply``
            (n-ary stages floor it). Defaults to the loader's
            ``load_batch_size`` when unset.
        outputs: Where the generated samples land (ledger + payload store).
        transformations: List of transformations, each applied independently
            to the same selection; order fixes only the output order. Each
            entry is validated against its algorithm's registered config
            schema.
    """

    model_config = ConfigDict(extra="forbid")

    storage: StorageConfig | None = None
    errors: ErrorsConfig | None = None
    transform_batch_size: int | None = Field(
        default=None,
        gt=0,
        description="Target transform batch; defaults to the loader's load_batch_size.",
    )
    outputs: SamplingOutputsConfig = Field(description="Ledger + payload store configuration.")
    transformations: list[TransformationConfig] = Field(
        default_factory=list,
        description="Transformations, each applied independently to the same "
        "selection; order fixes only the output order. Each entry is "
        "validated against its algorithm's registered config schema.",
    )

    @field_validator("transformations", mode="before")
    @classmethod
    def _resolve(cls, v: Any) -> Any:
        """Resolve each raw transformation entry against its registered schema.

        The YAML entries are plain dicts; this validator looks each ``type``
        up in the transformation registry, takes the algorithm's own config
        schema (``algorithm.Config``), and validates the entry against it, so
        required parameters and unknown fields are refused at config load.

        Args:
            v: The raw transformations list, or already-resolved instances.

        Returns:
            The list with every entry validated against its specific
            registered schema.

        Raises:
            ValueError: If a transformation type is unknown or an entry does
                not pass its algorithm's schema.
        """
        if not isinstance(v, list):
            return v
        from kcai_data_sampling_core.utils.registry import get_transformations_registry

        registry = get_transformations_registry()
        resolved: list[TransformationConfig] = []
        for entry in v:
            entry = dict(entry)
            algo = registry.get(entry.get("type", ""))
            if algo is None:
                raise ValueError(f"unknown transformation type {entry.get('type')!r}")
            config_cls = getattr(algo, "Config", TransformationConfig)
            resolved.append(config_cls.model_validate(entry))
        return resolved
