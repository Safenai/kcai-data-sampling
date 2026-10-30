"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RandomCropFromBordersTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RandomCropFromBorders(AlbumentationsUnary):
    """The ``random_crop_from_borders`` transformation (geometric family).

    Wraps upstream ``RandomCropFromBorders`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: crop_left, crop_right, crop_top, crop_bottom.
    """

    algorithm = "random_crop_from_borders"

    Config = RandomCropFromBordersTransformationConfig

    transform_class = A.RandomCropFromBorders

    geometry = "resize_back"
