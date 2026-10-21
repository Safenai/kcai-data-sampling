"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import ImageCompressionTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class ImageCompression(AlbumentationsUnary):
    """The ``image_compression`` transformation (compression family).

    Wraps upstream ``ImageCompression`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: compression_type, quality_range.
    """

    algorithm = "image_compression"

    Config = ImageCompressionTransformationConfig

    transform_class = A.ImageCompression

    enum_maps = {
        "compression_type": {"jpeg": "jpeg", "webp": "webp"},
    }
