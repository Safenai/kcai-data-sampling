"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import ColorJitterTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class ColorJitter(AlbumentationsUnary):
    """The ``color_jitter`` transformation (color family).

    Wraps upstream ``ColorJitter`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: brightness, contrast, saturation, hue.
    """

    algorithm = "color_jitter"

    Config = ColorJitterTransformationConfig

    transform_class = A.ColorJitter
