"""Image-specific subclasses of the generic core contracts.

The generic core leaves the sample's space open (``sample_axes``,
``value_range``) and defines generic base classes with generic methods; this
package holds the image specializations — :class:`ImageBatch`, and the
transformation classes — each implementing/overriding the generic contract and
adding image-specific behavior. I/O stays out: reads happen in the ``-job``
dataloaders and writes in the ``-job`` outputwriters; everything flows as
``(B, H, W, 4)`` uint8 RGBA arrays.
"""

from kcai_data_sampling_images.api.selection import ImageBatch

__all__ = ["ImageBatch"]