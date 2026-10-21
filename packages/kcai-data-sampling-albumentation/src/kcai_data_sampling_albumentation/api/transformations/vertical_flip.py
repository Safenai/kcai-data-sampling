"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import VerticalFlipTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class VerticalFlip(AlbumentationsUnary):
    """The ``vertical_flip`` transformation (dihedral family).

    Wraps upstream ``VerticalFlip`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: none.
    """

    algorithm = "vertical_flip"

    Config = VerticalFlipTransformationConfig

    transform_class = A.VerticalFlip
