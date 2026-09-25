"""The pydantic configuration schema (dqm-ml style).

All models are ``extra="forbid"``: a typo in a YAML config fails validation up
front instead of being silently ignored.
"""

from kcai_data_sampling_core.models.config import ColumnsConfig, JobConfig, TransformationConfig
from kcai_data_sampling_core.models.dataloaders import (
    DataLoaderConfig,
    DataLoadersConfig,
    FilterConfig,
    SamplePathConfig,
    TransformConfig,
)
from kcai_data_sampling_core.models.global_ import ComputeConfig, ErrorsConfig, ImageErrorsConfig, StorageConfig, TabularErrorsConfig
from kcai_data_sampling_core.models.interfaces import SamplingInterfaceConfig
from kcai_data_sampling_core.models.models import ModelRefConfig, ModelsConfig
from kcai_data_sampling_core.models.outputs import SamplingOutputsConfig
from kcai_data_sampling_core.models.sweep import SweepConfig

__all__ = [
    "ColumnsConfig",
    "ComputeConfig",
    "DataLoaderConfig",
    "DataLoadersConfig",
    "ErrorsConfig",
    "FilterConfig",
    "ImageErrorsConfig",
    "JobConfig",
    "ModelRefConfig",
    "ModelsConfig",
    "SamplePathConfig",
    "SamplingInterfaceConfig",
    "SamplingOutputsConfig",
    "StorageConfig",
    "SweepConfig",
    "TabularErrorsConfig",
    "TransformationConfig",
    "TransformConfig",
]