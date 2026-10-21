"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import PadIfNeededTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class PadIfNeeded(AlbumentationsUnary):
    """The ``pad_if_needed`` transformation (geometric family).

    Wraps upstream ``PadIfNeeded`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: min_height, min_width, position, border_mode, fill.
    """

    algorithm = "pad_if_needed"

    Config = PadIfNeededTransformationConfig

    transform_class = A.PadIfNeeded

    geometry = "crop_back"

    enum_maps = {
        "position": {
            "center": "center",
            "top_left": "top_left",
            "top_right": "top_right",
            "bottom_left": "bottom_left",
            "bottom_right": "bottom_right",
            "random": "random",
        },
        "border_mode": {"constant": 0, "replicate": 1, "reflect": 2, "wrap": 3, "reflect_101": 4},
    }
