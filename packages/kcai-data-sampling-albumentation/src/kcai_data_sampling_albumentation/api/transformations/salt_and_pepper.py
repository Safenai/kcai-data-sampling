"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import SaltAndPepperTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class SaltAndPepper(AlbumentationsUnary):
    """The ``salt_and_pepper`` transformation (noise family).

    Wraps upstream ``SaltAndPepper`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: amount, salt_vs_pepper.
    """

    algorithm = "salt_and_pepper"

    Config = SaltAndPepperTransformationConfig

    transform_class = A.SaltAndPepper
