"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import GaussianBlurTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class GaussianBlur(AlbumentationsUnary):
    """The ``gaussian_blur`` transformation (blur family).

    Wraps upstream ``GaussianBlur`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: sigma_limit.
    """

    algorithm = "gaussian_blur"

    Config = GaussianBlurTransformationConfig

    transform_class = A.GaussianBlur
