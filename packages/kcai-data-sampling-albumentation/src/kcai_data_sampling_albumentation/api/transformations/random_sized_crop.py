"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RandomSizedCropTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RandomSizedCrop(AlbumentationsUnary):
    """The ``random_sized_crop`` transformation (geometric family).

    Wraps upstream ``RandomSizedCrop`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: min_max_height, size, w2h_ratio, area_for_downscale, interpolation.
    """

    algorithm = "random_sized_crop"

    Config = RandomSizedCropTransformationConfig

    transform_class = A.RandomSizedCrop

    geometry = "resize_back"

    enum_maps = {
        "area_for_downscale": {"none": None, "image": "image", "image_mask": "image_mask"},
        "interpolation": {"nearest": 0, "bilinear": 1, "bicubic": 2, "area": 3, "lanczos": 4},
    }
