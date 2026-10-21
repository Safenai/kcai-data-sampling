"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RandomShadowTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RandomShadow(AlbumentationsUnary):
    """The ``random_shadow`` transformation (weather family).

    Wraps upstream ``RandomShadow`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: num_shadows_limit, shadow_dimension, shadow_intensity_range.
    """

    algorithm = "random_shadow"

    Config = RandomShadowTransformationConfig

    transform_class = A.RandomShadow
