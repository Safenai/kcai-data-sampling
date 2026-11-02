"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RandomSnowTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RandomSnow(AlbumentationsUnary):
    """The ``random_snow`` transformation (weather family).

    Wraps upstream ``RandomSnow`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: brightness_coeff, snow_point_range, method.
    """

    algorithm = "random_snow"

    Config = RandomSnowTransformationConfig

    transform_class = A.RandomSnow

    enum_maps = {
        "method": {"bleach": "bleach", "texture": "texture"},
    }
