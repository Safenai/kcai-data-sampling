"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import MorphologicalTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class Morphological(AlbumentationsUnary):
    """The ``morphological`` transformation (color family).

    Wraps upstream ``Morphological`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: scale, operation.
    """

    algorithm = "morphological"

    Config = MorphologicalTransformationConfig

    transform_class = A.Morphological

    enum_maps = {
        "operation": {"dilation": "dilation", "erosion": "erosion"},
    }
