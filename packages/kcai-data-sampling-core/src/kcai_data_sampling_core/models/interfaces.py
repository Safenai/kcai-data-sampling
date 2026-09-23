"""The transformations interface configuration (NEW relative to dqm-ml).

There is exactly one pipeline stage — transform samples — so dqm-ml's three
interfaces collapse into one ``transformations`` interface.
"""

from pydantic import BaseModel, ConfigDict, Field

from kcai_data_sampling_core.models.global_ import ErrorsConfig, StorageConfig
from kcai_data_sampling_core.models.outputs import SamplingOutputsConfig


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
    transformations: list[dict] = Field(
        default_factory=list,
        description="Transformations, each applied independently to the same "
        "selection; order fixes only the output order. Each is a "
        "{name, type, params...} dict.",
    )
