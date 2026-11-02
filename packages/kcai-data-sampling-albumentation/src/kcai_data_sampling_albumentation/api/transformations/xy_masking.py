"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import XyMaskingTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class XyMasking(AlbumentationsUnary):
    """The ``xy_masking`` transformation (dropout family).

    Wraps upstream ``XYMasking`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: num_masks_x, num_masks_y, mask_x_length, mask_y_length, fill.
    """

    algorithm = "xy_masking"

    Config = XyMaskingTransformationConfig

    transform_class = A.XYMasking
