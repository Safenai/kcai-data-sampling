"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import PixelDropoutTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class PixelDropout(AlbumentationsUnary):
    """The ``pixel_dropout`` transformation (dropout family).

    Wraps upstream ``PixelDropout`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: dropout_prob, per_channel, drop_value.
    """

    algorithm = "pixel_dropout"

    Config = PixelDropoutTransformationConfig

    transform_class = A.PixelDropout
