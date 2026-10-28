"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import GridDropoutTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class GridDropout(AlbumentationsUnary):
    """The ``grid_dropout`` transformation (dropout family).

    Wraps upstream ``GridDropout`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: ratio, unit_size_range, holes_number_xy, random_offset, fill.
    """

    algorithm = "grid_dropout"

    Config = GridDropoutTransformationConfig

    transform_class = A.GridDropout
