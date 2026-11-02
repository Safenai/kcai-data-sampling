"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RandomSunFlareTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RandomSunFlare(AlbumentationsUnary):
    """The ``random_sun_flare`` transformation (weather family).

    Wraps upstream ``RandomSunFlare`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: src_radius, angle_range, num_flare_circles_range, method.
    """

    algorithm = "random_sun_flare"

    Config = RandomSunFlareTransformationConfig

    transform_class = A.RandomSunFlare

    enum_maps = {
        "method": {"overlay": "overlay", "physics_based": "physics_based"},
    }
