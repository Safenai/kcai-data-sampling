"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import HueSaturationValueTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class HueSaturationValue(AlbumentationsUnary):
    """The ``hue_saturation_value`` transformation (color family).

    Wraps upstream ``HueSaturationValue`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: hue_shift_limit, sat_shift_limit, val_shift_limit.
    """

    algorithm = "hue_saturation_value"

    Config = HueSaturationValueTransformationConfig

    transform_class = A.HueSaturationValue
