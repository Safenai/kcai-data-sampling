"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import CropTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class Crop(AlbumentationsUnary):
    """The ``crop`` transformation (geometric family).

    Wraps upstream ``Crop`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: x_min, y_min, x_max, y_max, pad_if_needed, border_mode, fill.
    """

    algorithm = "crop"

    Config = CropTransformationConfig

    transform_class = A.Crop

    geometry = "resize_back"

    enum_maps = {
        "border_mode": {"constant": 0, "replicate": 1, "reflect": 2, "wrap": 3, "reflect_101": 4},
    }
