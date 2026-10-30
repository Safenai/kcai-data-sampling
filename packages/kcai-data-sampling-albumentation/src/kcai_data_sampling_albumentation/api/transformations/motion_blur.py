"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import MotionBlurTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class MotionBlur(AlbumentationsUnary):
    """The ``motion_blur`` transformation (blur family).

    Wraps upstream ``MotionBlur`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: blur_limit, allow_shifted, angle_range, direction_range.
    """

    algorithm = "motion_blur"

    Config = MotionBlurTransformationConfig

    transform_class = A.MotionBlur
