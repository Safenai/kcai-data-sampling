"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RandomScaleTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RandomScale(AlbumentationsUnary):
    """The ``random_scale`` transformation (geometric family).

    Wraps upstream ``RandomScale`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: scale_limit, interpolation.
    """

    algorithm = "random_scale"

    Config = RandomScaleTransformationConfig

    transform_class = A.RandomScale

    geometry = "resize_back"

    enum_maps = {
        "interpolation": {"nearest": 0, "bilinear": 1, "bicubic": 2, "area": 3, "lanczos": 4},
    }
