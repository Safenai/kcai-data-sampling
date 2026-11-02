"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import CoarseDropoutTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class CoarseDropout(AlbumentationsUnary):
    """The ``coarse_dropout`` transformation (dropout family).

    Wraps upstream ``CoarseDropout`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: num_holes_range, hole_height_range, hole_width_range, fill.
    """

    algorithm = "coarse_dropout"

    Config = CoarseDropoutTransformationConfig

    transform_class = A.CoarseDropout
