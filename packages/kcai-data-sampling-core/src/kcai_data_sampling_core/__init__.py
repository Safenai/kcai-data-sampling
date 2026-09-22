"""KCAI data-sampling core: the sample-generation interface and its registry.

    T : x ↦ x′

A transformation is one fully specified operation: algorithm + resolved
parameters + seed + model. One output row per generated sample, recording what
produced it and what the algorithm declares about itself — no judgement,
nothing about the annotation.

The package is deliberately generic: nothing here assumes images. `api/`
defines the interface in terms of ``sample_axes`` + ``dtype`` + ``value_range``;
`models/` holds the pydantic configuration schema (dqm-ml style); `utils/`
holds the plugin registry and the in-memory transformation runner. Nothing in
this package touches the disk.
"""

from kcai_data_sampling_core.api import (
    DataSelection,
    FAMILY_BY_ROLE,
    Output,
    Sample,
    TargetModel,
    ToolModel,
    Transformation,
    UnaryTransformation,
)
from kcai_data_sampling_core.utils import TransformationRunner
from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry

__all__ = [
    "FAMILY_BY_ROLE",
    "DataSelection",
    "Output",
    "PluginLoadedRegistry",
    "Sample",
    "TargetModel",
    "ToolModel",
    "Transformation",
    "TransformationRunner",
    "UnaryTransformation",
]