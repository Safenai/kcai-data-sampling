"""The interface: base classes, and the three dataclasses (sample, selection, row)."""

from kcai_data_sampling_core.api.n_ary import NAryTransformation
from kcai_data_sampling_core.api.record import Record
from kcai_data_sampling_core.api.selection import DataSelection, Sample
from kcai_data_sampling_core.api.transformation import FAMILY_BY_ROLE, Transformation
from kcai_data_sampling_core.api.unary import UnaryTransformation

__all__ = ["FAMILY_BY_ROLE", "DataSelection", "NAryTransformation", "Record", "Sample", "Transformation", "UnaryTransformation"]
