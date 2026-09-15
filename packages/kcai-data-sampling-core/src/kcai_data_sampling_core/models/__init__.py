"""Model adapters. The interface asks a model only for its role: a target
model exposes ``grad(x)``; a tool model produces content (``inpaint``)."""

from kcai_data_sampling_core.models.lama import LamaTool
from kcai_data_sampling_core.models.weights import weights_path
from kcai_data_sampling_core.models.yolo import YoloTarget

__all__ = ["LamaTool", "YoloTarget", "weights_path"]
