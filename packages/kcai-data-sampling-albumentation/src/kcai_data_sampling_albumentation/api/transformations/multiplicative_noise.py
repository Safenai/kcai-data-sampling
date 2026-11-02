"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import MultiplicativeNoiseTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class MultiplicativeNoise(AlbumentationsUnary):
    """The ``multiplicative_noise`` transformation (noise family).

    Wraps upstream ``MultiplicativeNoise`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: multiplier, per_channel, elementwise.
    """

    algorithm = "multiplicative_noise"

    Config = MultiplicativeNoiseTransformationConfig

    transform_class = A.MultiplicativeNoise
