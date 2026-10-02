"""Model and transformation subclasses owned by this package.

The generic contracts live in ``kcai_data_sampling_core.api``; this package
holds its own model adapter (``api/models/``) and its own transformation class
(``api/transformations/``), each implementing the generic core contract for
the LaMa inpainting tool. Nothing here reads or writes images: batches flow in
as ``(B, H, W, 4)`` uint8 RGBA arrays and the only disk I/O in the package is
the weight cache.
"""
