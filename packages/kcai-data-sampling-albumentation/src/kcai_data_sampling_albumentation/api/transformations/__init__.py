"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

from kcai_data_sampling_albumentation.api.transformations.additive_noise import AdditiveNoise
from kcai_data_sampling_albumentation.api.transformations.advanced_blur import AdvancedBlur
from kcai_data_sampling_albumentation.api.transformations.affine import Affine
from kcai_data_sampling_albumentation.api.transformations.auto_contrast import AutoContrast
from kcai_data_sampling_albumentation.api.transformations.blur import Blur
from kcai_data_sampling_albumentation.api.transformations.center_crop import CenterCrop
from kcai_data_sampling_albumentation.api.transformations.channel_dropout import ChannelDropout
from kcai_data_sampling_albumentation.api.transformations.channel_shuffle import ChannelShuffle
from kcai_data_sampling_albumentation.api.transformations.chromatic_aberration import ChromaticAberration
from kcai_data_sampling_albumentation.api.transformations.clahe import Clahe
from kcai_data_sampling_albumentation.api.transformations.coarse_dropout import CoarseDropout
from kcai_data_sampling_albumentation.api.transformations.color_jitter import ColorJitter
from kcai_data_sampling_albumentation.api.transformations.crop import Crop
from kcai_data_sampling_albumentation.api.transformations.d4 import D4
from kcai_data_sampling_albumentation.api.transformations.defocus import Defocus
from kcai_data_sampling_albumentation.api.transformations.downscale import Downscale
from kcai_data_sampling_albumentation.api.transformations.elastic_transform import ElasticTransform
from kcai_data_sampling_albumentation.api.transformations.equalize import Equalize
from kcai_data_sampling_albumentation.api.transformations.erasing import Erasing
from kcai_data_sampling_albumentation.api.transformations.fancy_pca import FancyPca
from kcai_data_sampling_albumentation.api.transformations.frequency_masking import FrequencyMasking
from kcai_data_sampling_albumentation.api.transformations.gauss_noise import GaussNoise
from kcai_data_sampling_albumentation.api.transformations.gaussian_blur import GaussianBlur
from kcai_data_sampling_albumentation.api.transformations.grid_distortion import GridDistortion
from kcai_data_sampling_albumentation.api.transformations.grid_dropout import GridDropout
from kcai_data_sampling_albumentation.api.transformations.grid_elastic_deform import GridElasticDeform
from kcai_data_sampling_albumentation.api.transformations.hue_saturation_value import HueSaturationValue
from kcai_data_sampling_albumentation.api.transformations.illumination import Illumination
from kcai_data_sampling_albumentation.api.transformations.image_compression import ImageCompression
from kcai_data_sampling_albumentation.api.transformations.invert_img import InvertImg
from kcai_data_sampling_albumentation.api.transformations.isonoise import Isonoise
from kcai_data_sampling_albumentation.api.transformations.median_blur import MedianBlur
from kcai_data_sampling_albumentation.api.transformations.morphological import Morphological
from kcai_data_sampling_albumentation.api.transformations.motion_blur import MotionBlur
from kcai_data_sampling_albumentation.api.transformations.multiplicative_noise import MultiplicativeNoise
from kcai_data_sampling_albumentation.api.transformations.optical_distortion import OpticalDistortion
from kcai_data_sampling_albumentation.api.transformations.pad import Pad
from kcai_data_sampling_albumentation.api.transformations.pad_if_needed import PadIfNeeded
from kcai_data_sampling_albumentation.api.transformations.perspective import Perspective
from kcai_data_sampling_albumentation.api.transformations.piecewise_affine import PiecewiseAffine
from kcai_data_sampling_albumentation.api.transformations.pixel_dropout import PixelDropout
from kcai_data_sampling_albumentation.api.transformations.planckian_jitter import PlanckianJitter
from kcai_data_sampling_albumentation.api.transformations.plasma_brightness_contrast import PlasmaBrightnessContrast
from kcai_data_sampling_albumentation.api.transformations.plasma_shadow import PlasmaShadow
from kcai_data_sampling_albumentation.api.transformations.posterize import Posterize
from kcai_data_sampling_albumentation.api.transformations.random_brightness_contrast import RandomBrightnessContrast
from kcai_data_sampling_albumentation.api.transformations.random_crop import RandomCrop
from kcai_data_sampling_albumentation.api.transformations.random_crop_from_borders import RandomCropFromBorders
from kcai_data_sampling_albumentation.api.transformations.random_fog import RandomFog
from kcai_data_sampling_albumentation.api.transformations.random_gamma import RandomGamma
from kcai_data_sampling_albumentation.api.transformations.random_grid_shuffle import RandomGridShuffle
from kcai_data_sampling_albumentation.api.transformations.random_rain import RandomRain
from kcai_data_sampling_albumentation.api.transformations.random_resized_crop import RandomResizedCrop
from kcai_data_sampling_albumentation.api.transformations.random_scale import RandomScale
from kcai_data_sampling_albumentation.api.transformations.random_shadow import RandomShadow
from kcai_data_sampling_albumentation.api.transformations.random_sized_crop import RandomSizedCrop
from kcai_data_sampling_albumentation.api.transformations.random_snow import RandomSnow
from kcai_data_sampling_albumentation.api.transformations.random_sun_flare import RandomSunFlare
from kcai_data_sampling_albumentation.api.transformations.random_tone_curve import RandomToneCurve
from kcai_data_sampling_albumentation.api.transformations.resize import Resize
from kcai_data_sampling_albumentation.api.transformations.rgb_shift import RgbShift
from kcai_data_sampling_albumentation.api.transformations.ringing_overshoot import RingingOvershoot
from kcai_data_sampling_albumentation.api.transformations.rotate import Rotate
from kcai_data_sampling_albumentation.api.transformations.safe_rotate import SafeRotate
from kcai_data_sampling_albumentation.api.transformations.salt_and_pepper import SaltAndPepper
from kcai_data_sampling_albumentation.api.transformations.sharpen import Sharpen
from kcai_data_sampling_albumentation.api.transformations.shift_scale_rotate import ShiftScaleRotate
from kcai_data_sampling_albumentation.api.transformations.shot_noise import ShotNoise
from kcai_data_sampling_albumentation.api.transformations.solarize import Solarize
from kcai_data_sampling_albumentation.api.transformations.spatter import Spatter
from kcai_data_sampling_albumentation.api.transformations.square_symmetry import SquareSymmetry
from kcai_data_sampling_albumentation.api.transformations.thin_plate_spline import ThinPlateSpline
from kcai_data_sampling_albumentation.api.transformations.unsharp_mask import UnsharpMask
from kcai_data_sampling_albumentation.api.transformations.vertical_flip import VerticalFlip
from kcai_data_sampling_albumentation.api.transformations.xy_masking import XyMasking
from kcai_data_sampling_albumentation.api.transformations.zoom_blur import ZoomBlur

__all__ = [
    "D4",
    "AdditiveNoise",
    "AdvancedBlur",
    "Affine",
    "AutoContrast",
    "Blur",
    "CenterCrop",
    "ChannelDropout",
    "ChannelShuffle",
    "ChromaticAberration",
    "Clahe",
    "CoarseDropout",
    "ColorJitter",
    "Crop",
    "Defocus",
    "Downscale",
    "ElasticTransform",
    "Equalize",
    "Erasing",
    "FancyPca",
    "FrequencyMasking",
    "GaussNoise",
    "GaussianBlur",
    "GridDistortion",
    "GridDropout",
    "GridElasticDeform",
    "HueSaturationValue",
    "Illumination",
    "ImageCompression",
    "InvertImg",
    "Isonoise",
    "MedianBlur",
    "Morphological",
    "MotionBlur",
    "MultiplicativeNoise",
    "OpticalDistortion",
    "Pad",
    "PadIfNeeded",
    "Perspective",
    "PiecewiseAffine",
    "PixelDropout",
    "PlanckianJitter",
    "PlasmaBrightnessContrast",
    "PlasmaShadow",
    "Posterize",
    "RandomBrightnessContrast",
    "RandomCrop",
    "RandomCropFromBorders",
    "RandomFog",
    "RandomGamma",
    "RandomGridShuffle",
    "RandomRain",
    "RandomResizedCrop",
    "RandomScale",
    "RandomShadow",
    "RandomSizedCrop",
    "RandomSnow",
    "RandomSunFlare",
    "RandomToneCurve",
    "Resize",
    "RgbShift",
    "RingingOvershoot",
    "Rotate",
    "SafeRotate",
    "SaltAndPepper",
    "Sharpen",
    "ShiftScaleRotate",
    "ShotNoise",
    "Solarize",
    "Spatter",
    "SquareSymmetry",
    "ThinPlateSpline",
    "UnsharpMask",
    "VerticalFlip",
    "XyMasking",
    "ZoomBlur",
]
