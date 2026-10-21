"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import SquareSymmetryTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class SquareSymmetry(AlbumentationsUnary):
    """The ``square_symmetry`` transformation (dihedral family).

    Wraps upstream ``SquareSymmetry`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: none.
    """

    algorithm = "square_symmetry"

    Config = SquareSymmetryTransformationConfig

    transform_class = A.SquareSymmetry
