"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RandomBrightnessContrastTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RandomBrightnessContrast(AlbumentationsUnary):
    """The ``random_brightness_contrast`` transformation (color family).

    Wraps upstream ``RandomBrightnessContrast`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: brightness_limit, contrast_limit, ensure_safe_range.
    """

    algorithm = "random_brightness_contrast"

    Config = RandomBrightnessContrastTransformationConfig

    transform_class = A.RandomBrightnessContrast
