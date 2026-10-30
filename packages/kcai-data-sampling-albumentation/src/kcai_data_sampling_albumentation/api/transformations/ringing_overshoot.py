"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import RingingOvershootTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class RingingOvershoot(AlbumentationsUnary):
    """The ``ringing_overshoot`` transformation (blur family).

    Wraps upstream ``RingingOvershoot`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: blur_limit, cutoff.
    """

    algorithm = "ringing_overshoot"

    Config = RingingOvershootTransformationConfig

    transform_class = A.RingingOvershoot
