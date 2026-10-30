"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RandomRainTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RandomRain(AlbumentationsUnary):
    """The ``random_rain`` transformation (weather family).

    Wraps upstream ``RandomRain`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: slant_range, drop_length, drop_width, blur_value, brightness_coefficient, rain_type.
    """

    algorithm = "random_rain"

    Config = RandomRainTransformationConfig

    transform_class = A.RandomRain

    enum_maps = {
        "rain_type": {"default": "default", "drizzle": "drizzle", "heavy": "heavy", "torrential": "torrential"},
    }
