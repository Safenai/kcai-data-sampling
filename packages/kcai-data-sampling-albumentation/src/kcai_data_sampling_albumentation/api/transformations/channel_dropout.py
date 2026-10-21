"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import ChannelDropoutTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class ChannelDropout(AlbumentationsUnary):
    """The ``channel_dropout`` transformation (color family).

    Wraps upstream ``ChannelDropout`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: channel_drop_range, fill.
    """

    algorithm = "channel_dropout"

    Config = ChannelDropoutTransformationConfig

    transform_class = A.ChannelDropout
