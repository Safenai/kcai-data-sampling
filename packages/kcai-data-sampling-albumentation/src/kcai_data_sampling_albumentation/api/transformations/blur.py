"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import BlurTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class Blur(AlbumentationsUnary):
    """The ``blur`` transformation (blur family).

    Wraps upstream ``Blur`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: blur_limit.
    """

    algorithm = "blur"

    Config = BlurTransformationConfig

    transform_class = A.Blur
