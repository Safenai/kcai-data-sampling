"""Unary image transformation math functions.

All image algorithms share the RGBA-U8 array convention ``(B, H, W, 4) uint8``
and the ``("height", "width", "channel")`` sample layout. The module-level
functions here are the pure algorithm math; the transformation classes that
wrap them into the unary contract live in
``kcai_data_sampling_images.api.transformations``.
"""

from kcai_data_sampling_images.transformations.crop_resize import crop_resize
from kcai_data_sampling_images.transformations.horizontal_flip import horizontal_flip

__all__ = ["crop_resize", "horizontal_flip"]