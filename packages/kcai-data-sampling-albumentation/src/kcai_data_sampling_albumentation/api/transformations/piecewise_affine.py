"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import PiecewiseAffineTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class PiecewiseAffine(AlbumentationsUnary):
    """The ``piecewise_affine`` transformation (geometric family).

    Wraps upstream ``PiecewiseAffine`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: scale, nb_rows, nb_cols, interpolation, border_mode, fill.
    """

    algorithm = "piecewise_affine"

    Config = PiecewiseAffineTransformationConfig

    transform_class = A.PiecewiseAffine

    enum_maps = {
        "interpolation": {"nearest": 0, "bilinear": 1, "bicubic": 2, "area": 3, "lanczos": 4},
        "border_mode": {"constant": 0, "replicate": 1, "reflect": 2, "wrap": 3, "reflect_101": 4},
    }
