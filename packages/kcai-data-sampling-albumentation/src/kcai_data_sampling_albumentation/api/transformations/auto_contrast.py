"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import AutoContrastTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class AutoContrast(AlbumentationsUnary):
    """The ``auto_contrast`` transformation (color family).

    Wraps upstream ``AutoContrast`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: cutoff, method.
    """

    algorithm = "auto_contrast"

    Config = AutoContrastTransformationConfig

    transform_class = A.AutoContrast

    enum_maps = {
        "method": {"cdf": "cdf", "pixel": "pixel"},
    }
