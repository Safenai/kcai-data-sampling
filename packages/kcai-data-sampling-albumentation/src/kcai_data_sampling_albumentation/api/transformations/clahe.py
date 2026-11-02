"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import ClaheTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class Clahe(AlbumentationsUnary):
    """The ``clahe`` transformation (color family).

    Wraps upstream ``CLAHE`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: clip_limit, tile_grid_size.
    """

    algorithm = "clahe"

    Config = ClaheTransformationConfig

    transform_class = A.CLAHE
