"""Unary image transformation classes.

The generic core contract lives in ``kcai_data_sampling_core.api``; these are
the image-specific subclasses, each wrapping a pure algorithm function from
``kcai_data_sampling_images.transformations`` into the unary ``apply``
contract.
"""

from kcai_data_sampling_images.api.transformations.crop_resize import CropResize
from kcai_data_sampling_images.api.transformations.horizontal_flip import HorizontalFlip

__all__ = ["CropResize", "HorizontalFlip"]