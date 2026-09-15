"""KCAI Data Sampling, the sample generation interface.

    T : x ↦ x′

A transformation is one fully specified operation: algorithm + resolved
parameters + seed + model. One row per output, recording what produced it and
what the algorithm declares about itself, no judgement, nothing about the
annotation.
"""

from kcai_data_sampling_core.api import DataSelection, NAryTransformation, Record, Sample, Transformation, UnaryTransformation
from kcai_data_sampling_core.transformations import FGSM, CropResize, CutMix, HorizontalFlip, Inpaint
from kcai_data_sampling_core.utils import TransformationRunner

__all__ = [
    "FGSM",
    "CropResize",
    "CutMix",
    "DataSelection",
    "HorizontalFlip",
    "Inpaint",
    "NAryTransformation",
    "Record",
    "Sample",
    "Transformation",
    "TransformationRunner",
    "UnaryTransformation",
]
