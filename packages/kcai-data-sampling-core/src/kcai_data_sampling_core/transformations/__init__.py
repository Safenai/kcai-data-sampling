"""Transformation algorithms: three procedural (one of them n-ary), one
generative, one adversarial."""

from kcai_data_sampling_core.transformations.crop_resize import CropResize
from kcai_data_sampling_core.transformations.cutmix import CutMix
from kcai_data_sampling_core.transformations.fgsm import FGSM
from kcai_data_sampling_core.transformations.horizontal_flip import HorizontalFlip
from kcai_data_sampling_core.transformations.inpaint import Inpaint

__all__ = ["FGSM", "CropResize", "CutMix", "HorizontalFlip", "Inpaint"]
