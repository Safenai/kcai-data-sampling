"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RandomGridShuffleTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RandomGridShuffle(AlbumentationsUnary):
    """The ``random_grid_shuffle`` transformation (dropout family).

    Wraps upstream ``RandomGridShuffle`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: grid.
    """

    algorithm = "random_grid_shuffle"

    Config = RandomGridShuffleTransformationConfig

    transform_class = A.RandomGridShuffle
