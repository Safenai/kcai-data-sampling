"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RgbShiftTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RgbShift(AlbumentationsUnary):
    """The ``rgb_shift`` transformation (color family).

    Wraps upstream ``RGBShift`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: r_shift_limit, g_shift_limit, b_shift_limit.
    """

    algorithm = "rgb_shift"

    Config = RgbShiftTransformationConfig

    transform_class = A.RGBShift
