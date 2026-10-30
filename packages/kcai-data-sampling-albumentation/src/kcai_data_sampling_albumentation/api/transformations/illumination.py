"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import IlluminationTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class Illumination(AlbumentationsUnary):
    """The ``illumination`` transformation (color family).

    Wraps upstream ``Illumination`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: mode, intensity_range, effect_type, angle_range, sigma_range.
    """

    algorithm = "illumination"

    Config = IlluminationTransformationConfig

    transform_class = A.Illumination

    enum_maps = {
        "mode": {"linear": "linear", "corner": "corner", "gaussian": "gaussian"},
        "effect_type": {"both": "both", "brighten": "brighten", "darken": "darken"},
    }
