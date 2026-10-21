"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import ZoomBlurTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class ZoomBlur(AlbumentationsUnary):
    """The ``zoom_blur`` transformation (blur family).

    Wraps upstream ``ZoomBlur`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: max_factor, step_factor.
    """

    algorithm = "zoom_blur"

    Config = ZoomBlurTransformationConfig

    transform_class = A.ZoomBlur
