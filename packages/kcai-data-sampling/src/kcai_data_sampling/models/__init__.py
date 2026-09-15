"""Model adapters. The interface asks a model only for its role: a target
model exposes ``grad(x)``; a tool model produces content."""

from kcai_data_sampling.models.yolo import YoloTarget

__all__ = ["YoloTarget"]
