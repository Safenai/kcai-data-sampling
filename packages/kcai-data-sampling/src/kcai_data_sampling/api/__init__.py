"""The interface: base classes, and the three dataclasses (sample, selection, row)."""

from kcai_data_sampling.api.n_ary import NAryTransformation
from kcai_data_sampling.api.record import Record
from kcai_data_sampling.api.selection import DataSelection, Sample
from kcai_data_sampling.api.transformation import FAMILY_BY_ROLE, Transformation
from kcai_data_sampling.api.unary import UnaryTransformation

__all__ = ["FAMILY_BY_ROLE", "DataSelection", "NAryTransformation", "Record", "Sample", "Transformation", "UnaryTransformation"]
