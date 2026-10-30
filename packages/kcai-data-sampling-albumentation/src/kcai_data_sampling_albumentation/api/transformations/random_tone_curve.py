"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RandomToneCurveTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RandomToneCurve(AlbumentationsUnary):
    """The ``random_tone_curve`` transformation (color family).

    Wraps upstream ``RandomToneCurve`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: scale, per_channel.
    """

    algorithm = "random_tone_curve"

    Config = RandomToneCurveTransformationConfig

    transform_class = A.RandomToneCurve
