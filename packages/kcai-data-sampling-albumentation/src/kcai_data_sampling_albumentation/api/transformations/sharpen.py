"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import SharpenTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class Sharpen(AlbumentationsUnary):
    """The ``sharpen`` transformation (blur family).

    Wraps upstream ``Sharpen`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: alpha, lightness.
    """

    algorithm = "sharpen"

    Config = SharpenTransformationConfig

    transform_class = A.Sharpen
