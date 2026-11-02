"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import UnsharpMaskTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class UnsharpMask(AlbumentationsUnary):
    """The ``unsharp_mask`` transformation (blur family).

    Wraps upstream ``UnsharpMask`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: blur_limit, sigma_limit, alpha, threshold.
    """

    algorithm = "unsharp_mask"

    Config = UnsharpMaskTransformationConfig

    transform_class = A.UnsharpMask
