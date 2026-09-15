"""Model adapters. A tool model is the package's choice (``LamaTool``). A
target model is the user's: any object satisfying ``api.roles.TargetModel``,
of which ``YoloTarget`` is one example, kept for the tests and the notebook."""

from kcai_data_sampling_core.models.lama import LamaTool
from kcai_data_sampling_core.models.weights import weights_path
from kcai_data_sampling_core.models.yolo import YoloTarget

__all__ = ["LamaTool", "YoloTarget", "weights_path"]
