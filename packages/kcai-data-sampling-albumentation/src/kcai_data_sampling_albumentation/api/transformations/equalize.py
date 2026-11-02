"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import EqualizeTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class Equalize(AlbumentationsUnary):
    """The ``equalize`` transformation (color family).

    Wraps upstream ``Equalize`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: mode, by_channels.
    """

    algorithm = "equalize"

    Config = EqualizeTransformationConfig

    transform_class = A.Equalize

    enum_maps = {
        "mode": {"cv": "cv", "pil": "pil"},
    }
