"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import PlanckianJitterTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class PlanckianJitter(AlbumentationsUnary):
    """The ``planckian_jitter`` transformation (color family).

    Wraps upstream ``PlanckianJitter`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: mode, sampling_method, temperature_limit.
    """

    algorithm = "planckian_jitter"

    Config = PlanckianJitterTransformationConfig

    transform_class = A.PlanckianJitter

    enum_maps = {
        "mode": {"blackbody": "blackbody", "cied": "cied"},
        "sampling_method": {"uniform": "uniform", "gaussian": "gaussian"},
    }
