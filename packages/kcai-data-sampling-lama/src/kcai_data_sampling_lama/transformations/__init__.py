"""Transformation math functions for this package.

The module-level function here (``transformations.inpaint.build_region_mask``)
is the pure mask math; the transformation class that wraps it into the unary
contract lives in ``kcai_data_sampling_lama.api.transformations``. All
batches share the RGBA-U8 array convention ``(B, H, W, 4) uint8``.
"""

from kcai_data_sampling_lama.transformations.inpaint import build_region_mask

__all__ = ["build_region_mask"]
