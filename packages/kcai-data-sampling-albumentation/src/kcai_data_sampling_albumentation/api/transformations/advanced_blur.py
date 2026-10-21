"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import AdvancedBlurTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class AdvancedBlur(AlbumentationsUnary):
    """The ``advanced_blur`` transformation (blur family).

    Wraps upstream ``AdvancedBlur`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: blur_limit, sigma_x_limit, sigma_y_limit, rotate_limit.
    """

    algorithm = "advanced_blur"

    Config = AdvancedBlurTransformationConfig

    transform_class = A.AdvancedBlur
