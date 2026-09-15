"""The interface: base classes, and the three dataclasses (sample, selection, output). Nothing here touches the disk."""

from kcai_data_sampling_core.api.n_ary import NAryTransformation
from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_core.api.roles import TargetModel, ToolModel
from kcai_data_sampling_core.api.selection import DataSelection, Sample
from kcai_data_sampling_core.api.transformation import FAMILY_BY_ROLE, Transformation
from kcai_data_sampling_core.api.unary import UnaryTransformation

__all__ = ["FAMILY_BY_ROLE", "DataSelection", "NAryTransformation", "Output", "Sample", "TargetModel", "ToolModel", "Transformation", "UnaryTransformation"]
