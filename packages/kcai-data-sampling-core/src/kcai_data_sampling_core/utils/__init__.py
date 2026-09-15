"""The runner, and the image codec the readers use."""

from kcai_data_sampling_core.utils.images import load_image, quantize, save_image
from kcai_data_sampling_core.utils.runner import TransformationRunner

__all__ = ["TransformationRunner", "load_image", "quantize", "save_image"]
