"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import PlasmaBrightnessContrastTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class PlasmaBrightnessContrast(AlbumentationsUnary):
    """The ``plasma_brightness_contrast`` transformation (color family).

    Wraps upstream ``PlasmaBrightnessContrast`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: brightness_range, contrast_range, plasma_size, roughness.
    """

    algorithm = "plasma_brightness_contrast"

    Config = PlasmaBrightnessContrastTransformationConfig

    transform_class = A.PlasmaBrightnessContrast
