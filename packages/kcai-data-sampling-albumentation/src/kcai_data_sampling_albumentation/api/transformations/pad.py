"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import PadTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class Pad(AlbumentationsUnary):
    """The ``pad`` transformation (geometric family).

    Wraps upstream ``Pad`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: padding, border_mode, fill.
    """

    algorithm = "pad"

    Config = PadTransformationConfig

    transform_class = A.Pad

    geometry = "crop_back"

    enum_maps = {
        "border_mode": {"constant": 0, "replicate": 1, "reflect": 2, "wrap": 3, "reflect_101": 4},
    }
