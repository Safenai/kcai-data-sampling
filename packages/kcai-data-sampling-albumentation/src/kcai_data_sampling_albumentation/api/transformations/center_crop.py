"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import CenterCropTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class CenterCrop(AlbumentationsUnary):
    """The ``center_crop`` transformation (geometric family).

    Wraps upstream ``CenterCrop`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: height, width, pad_if_needed, border_mode, fill.
    """

    algorithm = "center_crop"

    Config = CenterCropTransformationConfig

    transform_class = A.CenterCrop

    geometry = "resize_back"

    enum_maps = {
        "border_mode": {"constant": 0, "replicate": 1, "reflect": 2, "wrap": 3, "reflect_101": 4},
    }
