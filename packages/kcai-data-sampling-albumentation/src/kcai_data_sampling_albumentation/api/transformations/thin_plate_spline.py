"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import ThinPlateSplineTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class ThinPlateSpline(AlbumentationsUnary):
    """The ``thin_plate_spline`` transformation (geometric family).

    Wraps upstream ``ThinPlateSpline`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: scale_range, num_control_points, interpolation, border_mode, fill.
    """

    algorithm = "thin_plate_spline"

    Config = ThinPlateSplineTransformationConfig

    transform_class = A.ThinPlateSpline

    enum_maps = {
        "interpolation": {"nearest": 0, "bilinear": 1, "bicubic": 2, "area": 3, "lanczos": 4},
        "border_mode": {"constant": 0, "replicate": 1, "reflect": 2, "wrap": 3, "reflect_101": 4},
    }
