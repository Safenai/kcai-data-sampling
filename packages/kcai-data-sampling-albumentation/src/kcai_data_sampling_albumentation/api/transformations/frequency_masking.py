"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

import albumentations as A  # noqa: N812

from kcai_data_sampling_albumentation.configs import FrequencyMaskingTransformationConfig
from kcai_data_sampling_albumentation.engine import AlbumentationsUnary


class FrequencyMasking(AlbumentationsUnary):
    """The ``frequency_masking`` transformation (dropout family).

    Wraps upstream ``FrequencyMasking`` via the shared engine: per-row re-seeding,
    alpha-split for image-only transforms, uint8 shape contract.
    Exposed parameters: freq_mask_param.
    """

    algorithm = "frequency_masking"

    Config = FrequencyMaskingTransformationConfig

    transform_class = A.FrequencyMasking
