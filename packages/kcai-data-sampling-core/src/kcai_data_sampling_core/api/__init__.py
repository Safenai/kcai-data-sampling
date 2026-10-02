"""The generic sample-transformation interface.

Nothing here touches the disk, and nothing here assumes images: a sample is
defined by its axes, its dtype and its value range (see ``selection.py``), and
the image packages specialize that contract (there ``x`` is ``(H, W, C)``).
The loader/writer contracts (``DataLoader``, ``DataSelection``,
``OutputWriter``) are generic too; their datatype-specific subclasses live in
the packages' own ``api/`` folders.
"""

from kcai_data_sampling_core.api.dataloaders import DataLoader, DataSelection
from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_core.api.output_writer import OutputWriter
from kcai_data_sampling_core.api.roles import TargetModel, ToolModel
from kcai_data_sampling_core.api.selection import Batch
from kcai_data_sampling_core.api.transformation import FAMILY_BY_ROLE, Transformation
from kcai_data_sampling_core.api.unary import UnaryTransformation

__all__ = [
    "FAMILY_BY_ROLE",
    "Batch",
    "DataLoader",
    "DataSelection",
    "Output",
    "OutputWriter",
    "TargetModel",
    "ToolModel",
    "Transformation",
    "UnaryTransformation",
]
