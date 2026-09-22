"""Unary image transformations: mirror, resize/crop.

All image algorithms share the RGBA-U8 array convention
``(B, H, W, 4) uint8`` and the ``("height", "width", "channel")`` sample
layout.
"""

from kcai_data_sampling_core.api.transformation import (
    Transformation,
)
from kcai_data_sampling_core.api.unary import UnaryTransformation
from kcai_data_sampling_core.utils.registry import get_transformations_registry
from kcai_data_sampling_images.configs import (
    CropResizeTransformationConfig,
    HorizontalFlipTransformationConfig,
)

__all__ = ["Transformation", "UnaryTransformation", "get_transformations_registry"]