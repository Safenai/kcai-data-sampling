"""Named model references (NEW relative to dqm-ml).

The YAML ``models:`` section names reusable tool / target models; a
transformation references one by name (``target_model: yolo``) and the job
resolves the name to an instance via the models registry, so the YAML never
touches Python objects.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelRefConfig(BaseModel):
    """A named reference to a registered model plugin.

    Attributes:
        type: Registered ``kcai_data_sampling.models`` plugin (a ``ToolModel``
            or ``TargetModel`` adapter).
        weights: Optional weights/checkpoint the plugin may cache.
        params: Plugin-specific knobs.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Registered model plugin name.")
    weights: str | None = Field(default=None, description="Weights/checkpoint file name.")
    params: dict[str, Any] = Field(default_factory=dict, description="Plugin-specific knobs.")


class ModelsConfig(BaseModel):
    """Named collection of model references.

    Attributes:
        models: Mapping from model name to a model reference; transformations
            reference models by these names.
    """

    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelRefConfig] = Field(description="Named model references.")