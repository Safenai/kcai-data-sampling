"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import GaussNoiseTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class GaussNoise(AlbumentationsUnary):
    """The ``gauss_noise`` transformation (noise family).

    Wraps upstream ``GaussNoise`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: std_range, mean_range, per_channel, noise_scale_factor.
    """

    algorithm = "gauss_noise"

    Config = GaussNoiseTransformationConfig

    transform_class = A.GaussNoise
