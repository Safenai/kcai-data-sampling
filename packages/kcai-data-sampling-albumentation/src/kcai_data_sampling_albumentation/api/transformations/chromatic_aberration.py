"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import ChromaticAberrationTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class ChromaticAberration(AlbumentationsUnary):
    """The ``chromatic_aberration`` transformation (color family).

    Wraps upstream ``ChromaticAberration`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: primary_distortion_limit, secondary_distortion_limit, mode, interpolation.
    """

    algorithm = "chromatic_aberration"

    Config = ChromaticAberrationTransformationConfig

    transform_class = A.ChromaticAberration

    enum_maps = {
        "mode": {"green_purple": "green_purple", "red_blue": "red_blue", "random": "random"},
        "interpolation": {"nearest": 0, "bilinear": 1, "bicubic": 2, "area": 3, "lanczos": 4},
    }
