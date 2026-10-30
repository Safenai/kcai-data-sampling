"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import ShiftScaleRotateTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class ShiftScaleRotate(AlbumentationsUnary):
    """The ``shift_scale_rotate`` transformation (geometric family).

    Wraps upstream ``ShiftScaleRotate`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: shift_limit, shift_limit_x, shift_limit_y, scale_limit, rotate_limit, interpolation,
    border_mode, fill.
    """

    algorithm = "shift_scale_rotate"

    Config = ShiftScaleRotateTransformationConfig

    transform_class = A.ShiftScaleRotate

    enum_maps = {
        "interpolation": {"nearest": 0, "bilinear": 1, "bicubic": 2, "area": 3, "lanczos": 4},
        "border_mode": {"constant": 0, "replicate": 1, "reflect": 2, "wrap": 3, "reflect_101": 4},
    }
