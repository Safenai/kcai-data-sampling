"""The package's tool models (``LamaTool``) and their weights cache. Target
models are the user's and live outside: any object satisfying
``api.roles.TargetModel`` (``examples/notebooks/yolo_target.py`` is one)."""

from kcai_data_sampling_core.models.lama import LamaTool
from kcai_data_sampling_core.models.weights import weights_path

__all__ = ["LamaTool", "weights_path"]
