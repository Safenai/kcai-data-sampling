"""Model adapters. The interface asks a model only for its role: a target
model exposes ``grad(x)``; a tool model produces content (``inpaint``)."""

from kcai_data_sampling.models.lama import LamaTool
from kcai_data_sampling.models.weights import weights_path
from kcai_data_sampling.models.yolo import YoloTarget

__all__ = ["LamaTool", "YoloTarget", "weights_path"]
