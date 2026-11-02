"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import FancyPcaTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class FancyPca(AlbumentationsUnary):
    """The ``fancy_pca`` transformation (color family).

    Wraps upstream ``FancyPCA`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: alpha.
    """

    algorithm = "fancy_pca"

    Config = FancyPcaTransformationConfig

    transform_class = A.FancyPCA
