"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import PerspectiveTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class Perspective(AlbumentationsUnary):
    """The ``perspective`` transformation (geometric family).

    Wraps upstream ``Perspective`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: scale, interpolation, border_mode, fill.
    """

    algorithm = "perspective"

    Config = PerspectiveTransformationConfig

    transform_class = A.Perspective

    pinned = {"keep_size": True, "fit_output": False}

    enum_maps = {
        "interpolation": {"nearest": 0, "bilinear": 1, "bicubic": 2, "area": 3, "lanczos": 4},
        "border_mode": {"constant": 0, "replicate": 1, "reflect": 2, "wrap": 3, "reflect_101": 4},
    }
