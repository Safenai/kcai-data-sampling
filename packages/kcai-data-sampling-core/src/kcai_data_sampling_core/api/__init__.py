"""The generic sample-transformation interface.

Nothing here touches the disk, and nothing here assumes images: a sample is
defined by its axes, its dtype and its value range (see ``selection.py``), and
the image packages specialize that contract (there ``x`` is ``(H, W, C)``).
"""

from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_core.api.roles import TargetModel, ToolModel
from kcai_data_sampling_core.api.selection import DataSelection, Sample
from kcai_data_sampling_core.api.transformation import FAMILY_BY_ROLE, Transformation
from kcai_data_sampling_core.api.unary import UnaryTransformation

__all__ = [
    "FAMILY_BY_ROLE",
    "DataSelection",
    "Output",
    "Sample",
    "TargetModel",
    "ToolModel",
    "Transformation",
    "UnaryTransformation",
]