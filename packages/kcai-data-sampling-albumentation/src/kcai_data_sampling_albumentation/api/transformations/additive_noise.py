"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import AdditiveNoiseTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class AdditiveNoise(AlbumentationsUnary):
    """The ``additive_noise`` transformation (noise family).

    Wraps upstream ``AdditiveNoise`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: noise_type, approximation.
    """

    algorithm = "additive_noise"

    Config = AdditiveNoiseTransformationConfig

    transform_class = A.AdditiveNoise

    enum_maps = {
        "noise_type": {"uniform": "uniform", "gaussian": "gaussian"},
    }
