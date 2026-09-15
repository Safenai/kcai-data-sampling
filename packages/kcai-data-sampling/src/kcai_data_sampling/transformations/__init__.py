"""Transformation algorithms: three procedural (one of them n-ary), one
generative, one adversarial."""

from kcai_data_sampling.transformations.crop_resize import CropResize
from kcai_data_sampling.transformations.cutmix import CutMix
from kcai_data_sampling.transformations.fgsm import FGSM
from kcai_data_sampling.transformations.horizontal_flip import HorizontalFlip
from kcai_data_sampling.transformations.inpaint import Inpaint

__all__ = ["FGSM", "CropResize", "CutMix", "HorizontalFlip", "Inpaint"]
