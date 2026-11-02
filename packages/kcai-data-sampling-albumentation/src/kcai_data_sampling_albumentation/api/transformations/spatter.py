"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import SpatterTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class Spatter(AlbumentationsUnary):
    """The ``spatter`` transformation (weather family).

    Wraps upstream ``Spatter`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: mean, std, gauss_sigma, cutout_threshold, intensity, mode.
    """

    algorithm = "spatter"

    Config = SpatterTransformationConfig

    transform_class = A.Spatter

    enum_maps = {
        "mode": {"rain": "rain", "mud": "mud"},
    }
