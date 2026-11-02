"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import PlasmaShadowTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class PlasmaShadow(AlbumentationsUnary):
    """The ``plasma_shadow`` transformation (color family).

    Wraps upstream ``PlasmaShadow`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: shadow_intensity_range, plasma_size, roughness.
    """

    algorithm = "plasma_shadow"

    Config = PlasmaShadowTransformationConfig

    transform_class = A.PlasmaShadow
