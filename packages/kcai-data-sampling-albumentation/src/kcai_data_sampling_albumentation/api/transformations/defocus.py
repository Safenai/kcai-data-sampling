"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import DefocusTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class Defocus(AlbumentationsUnary):
    """The ``defocus`` transformation (blur family).

    Wraps upstream ``Defocus`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: radius, alias_blur.
    """

    algorithm = "defocus"

    Config = DefocusTransformationConfig

    transform_class = A.Defocus
