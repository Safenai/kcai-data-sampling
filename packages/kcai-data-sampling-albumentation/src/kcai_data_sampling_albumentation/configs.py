"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.

One config schema / transformation class per allowlisted upstream transform.
Regenerate with `python -m kcai_data_sampling_albumentation._generate`.
"""

from typing import Literal, Self

from kcai_data_sampling_core.models.sweep import SweepConfig
from pydantic import Field, model_validator

from kcai_data_sampling_albumentation.config_base import AlbumentationsConfig


class VerticalFlipTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``vertical_flip`` transformation."""

    type: Literal["vertical_flip"] = "vertical_flip"

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on vertical_flip parameters, value or sweep."""
        self._check_p()
        return self


class RotateTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``rotate`` transformation."""

    type: Literal["rotate"] = "rotate"
    limit: list[float] | float | SweepConfig = Field(
        default=[-90.0, 90.0],
        description="limit (range_pair_float); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on rotate parameters, value or sweep."""
        self._check_range_pair("limit", minimum=-360.0, maximum=360.0)
        self._check_p()
        return self


class ShiftScaleRotateTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``shift_scale_rotate`` transformation."""

    type: Literal["shift_scale_rotate"] = "shift_scale_rotate"
    shift_limit: list[float] | float | SweepConfig = Field(
        default=[-0.0625, 0.0625],
        description="shift_limit (range_pair_float); a SweepConfig expands it.",
    )
    shift_limit_x: list[float] | float | SweepConfig | None = Field(
        default=None,
        description="shift_limit_x (range_pair_float); a SweepConfig expands it.",
    )
    shift_limit_y: list[float] | float | SweepConfig | None = Field(
        default=None,
        description="shift_limit_y (range_pair_float); a SweepConfig expands it.",
    )
    scale_limit: list[float] | float | SweepConfig = Field(
        default=[-0.1, 0.1],
        description="scale_limit (range_pair_float); a SweepConfig expands it.",
    )
    rotate_limit: list[float] | float | SweepConfig = Field(
        default=[-45.0, 45.0],
        description="rotate_limit (range_pair_float); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on shift_scale_rotate parameters, value or sweep."""
        self._check_range_pair("shift_limit", minimum=-1.0, maximum=1.0)
        self._check_range_pair("shift_limit_x", minimum=-1.0, maximum=1.0)
        self._check_range_pair("shift_limit_y", minimum=-1.0, maximum=1.0)
        self._check_range_pair("scale_limit", minimum=-1.0)
        self._check_range_pair("rotate_limit", minimum=-360.0, maximum=360.0)
        self._check_p()
        return self


class ElasticTransformTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``elastic_transform`` transformation."""

    type: Literal["elastic_transform"] = "elastic_transform"
    alpha: float | int | SweepConfig = Field(
        default=1,
        description="alpha (scalar); a SweepConfig expands it.",
    )
    sigma: float | int | SweepConfig = Field(
        default=50,
        description="sigma (scalar); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on elastic_transform parameters, value or sweep."""
        self._check_parameter_bounds("alpha", minimum=0.0)
        self._check_parameter_bounds("sigma", minimum=1.0)
        self._check_p()
        return self


class MedianBlurTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``median_blur`` transformation."""

    type: Literal["median_blur"] = "median_blur"
    blur_limit: list[int] | int | SweepConfig = Field(
        default=[3, 7],
        description="blur_limit (range_pair_int); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on median_blur parameters, value or sweep."""
        self._check_range_pair("blur_limit", minimum=3, maximum=255)
        self._check_p()
        return self


class MotionBlurTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``motion_blur`` transformation."""

    type: Literal["motion_blur"] = "motion_blur"
    blur_limit: list[int] | int | SweepConfig = Field(
        default=[3, 7],
        description="blur_limit (range_pair_int); a SweepConfig expands it.",
    )
    allow_shifted: bool = Field(
        default=True,
        description="allow_shifted (bool); a SweepConfig expands it.",
    )
    angle_range: list[float] | float | SweepConfig = Field(
        default=[0.0, 360.0],
        description="angle_range (range_pair_float); a SweepConfig expands it.",
    )
    direction_range: list[float] | float | SweepConfig = Field(
        default=[-1.0, 1.0],
        description="direction_range (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on motion_blur parameters, value or sweep."""
        self._check_range_pair("blur_limit", minimum=3, maximum=255)
        self._check_range_pair("angle_range", minimum=0.0, maximum=360.0)
        self._check_range_pair("direction_range", minimum=-1.0, maximum=1.0)
        self._check_p()
        return self


class UnsharpMaskTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``unsharp_mask`` transformation."""

    type: Literal["unsharp_mask"] = "unsharp_mask"
    blur_limit: list[int] | int | SweepConfig = Field(
        default=[3, 7],
        description="blur_limit (range_pair_int); a SweepConfig expands it.",
    )
    sigma_limit: float | int | SweepConfig = Field(
        default=0.0,
        description="sigma_limit (scalar); a SweepConfig expands it.",
    )
    alpha: list[float] | float | SweepConfig = Field(
        default=[0.2, 0.5],
        description="alpha (range_pair_float); a SweepConfig expands it.",
    )
    threshold: float | int | SweepConfig = Field(
        default=10,
        description="threshold (scalar); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on unsharp_mask parameters, value or sweep."""
        self._check_range_pair("blur_limit", minimum=3, maximum=255)
        self._check_parameter_bounds("sigma_limit", minimum=0.0)
        self._check_range_pair("alpha", minimum=0.0, maximum=1.0)
        self._check_parameter_bounds("threshold", minimum=0, maximum=255)
        self._check_p()
        return self


class GaussNoiseTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``gauss_noise`` transformation."""

    type: Literal["gauss_noise"] = "gauss_noise"
    std_range: list[float] | float | SweepConfig = Field(
        default=[0.2, 0.44],
        description="std_range (range_pair_float); a SweepConfig expands it.",
    )
    mean_range: list[float] | float | SweepConfig = Field(
        default=[0.0, 0.0],
        description="mean_range (range_pair_float); a SweepConfig expands it.",
    )
    per_channel: bool = Field(
        default=True,
        description="per_channel (bool); a SweepConfig expands it.",
    )
    noise_scale_factor: float | int | SweepConfig = Field(
        default=1,
        description="noise_scale_factor (scalar); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on gauss_noise parameters, value or sweep."""
        self._check_range_pair("std_range", minimum=0.0, maximum=1.0)
        self._check_range_pair("mean_range", minimum=-1.0, maximum=1.0)
        self._check_parameter_bounds("noise_scale_factor", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class SaltAndPepperTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``salt_and_pepper`` transformation."""

    type: Literal["salt_and_pepper"] = "salt_and_pepper"
    amount: list[float] | float | SweepConfig = Field(
        default=[0.01, 0.06],
        description="amount (range_pair_float); a SweepConfig expands it.",
    )
    salt_vs_pepper: list[float] | float | SweepConfig = Field(
        default=[0.4, 0.6],
        description="salt_vs_pepper (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on salt_and_pepper parameters, value or sweep."""
        self._check_range_pair("amount", minimum=0.0, maximum=1.0)
        self._check_range_pair("salt_vs_pepper", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class ColorJitterTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``color_jitter`` transformation."""

    type: Literal["color_jitter"] = "color_jitter"
    brightness: list[float] | float | SweepConfig = Field(
        default=[0.8, 1.2],
        description="brightness (range_pair_float); a SweepConfig expands it.",
    )
    contrast: list[float] | float | SweepConfig = Field(
        default=[0.8, 1.2],
        description="contrast (range_pair_float); a SweepConfig expands it.",
    )
    saturation: list[float] | float | SweepConfig = Field(
        default=[0.8, 1.2],
        description="saturation (range_pair_float); a SweepConfig expands it.",
    )
    hue: list[float] | float | SweepConfig = Field(
        default=[-0.5, 0.5],
        description="hue (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on color_jitter parameters, value or sweep."""
        self._check_range_pair("brightness", minimum=0.0)
        self._check_range_pair("contrast", minimum=0.0)
        self._check_range_pair("saturation", minimum=0.0)
        self._check_range_pair("hue", minimum=-0.5, maximum=0.5)
        self._check_p()
        return self


class HueSaturationValueTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``hue_saturation_value`` transformation."""

    type: Literal["hue_saturation_value"] = "hue_saturation_value"
    hue_shift_limit: list[int] | int | SweepConfig = Field(
        default=[-20, 20],
        description="hue_shift_limit (range_pair_int); a SweepConfig expands it.",
    )
    sat_shift_limit: list[int] | int | SweepConfig = Field(
        default=[-30, 30],
        description="sat_shift_limit (range_pair_int); a SweepConfig expands it.",
    )
    val_shift_limit: list[int] | int | SweepConfig = Field(
        default=[-20, 20],
        description="val_shift_limit (range_pair_int); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on hue_saturation_value parameters, value or sweep."""
        self._check_range_pair("hue_shift_limit", minimum=-180, maximum=180)
        self._check_range_pair("sat_shift_limit", minimum=-255, maximum=255)
        self._check_range_pair("val_shift_limit", minimum=-255, maximum=255)
        self._check_p()
        return self


class RandomGammaTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_gamma`` transformation."""

    type: Literal["random_gamma"] = "random_gamma"
    gamma_limit: list[int] | int | SweepConfig = Field(
        default=[80, 120],
        description="gamma_limit (range_pair_int); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_gamma parameters, value or sweep."""
        self._check_range_pair("gamma_limit", minimum=1)
        self._check_p()
        return self


class SolarizeTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``solarize`` transformation."""

    type: Literal["solarize"] = "solarize"
    threshold_range: list[float] | float | SweepConfig = Field(
        default=[0.5, 0.5],
        description="threshold_range (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on solarize parameters, value or sweep."""
        self._check_range_pair("threshold_range", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class EqualizeTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``equalize`` transformation."""

    type: Literal["equalize"] = "equalize"
    mode: Literal["cv", "pil"] = Field(
        default="cv",
        description="mode (enum); a SweepConfig expands it.",
    )
    by_channels: bool = Field(
        default=True,
        description="by_channels (bool); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on equalize parameters, value or sweep."""
        self._check_p()
        return self


class CoarseDropoutTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``coarse_dropout`` transformation."""

    type: Literal["coarse_dropout"] = "coarse_dropout"
    num_holes_range: list[int] | int | SweepConfig = Field(
        default=[1, 2],
        description="num_holes_range (range_pair_int); a SweepConfig expands it.",
    )
    hole_height_range: list[float] | float | SweepConfig = Field(
        default=[0.1, 0.2],
        description="hole_height_range (range_pair_float); a SweepConfig expands it.",
    )
    hole_width_range: list[float] | float | SweepConfig = Field(
        default=[0.1, 0.2],
        description="hole_width_range (range_pair_float); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on coarse_dropout parameters, value or sweep."""
        self._check_range_pair("num_holes_range", minimum=1)
        self._check_range_pair("hole_height_range", minimum=0.0)
        self._check_range_pair("hole_width_range", minimum=0.0)
        self._check_p()
        return self


class ErasingTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``erasing`` transformation."""

    type: Literal["erasing"] = "erasing"
    scale: list[float] | float | SweepConfig = Field(
        default=[0.02, 0.33],
        description="scale (range_pair_float); a SweepConfig expands it.",
    )
    ratio: list[float] | float | SweepConfig = Field(
        default=[0.3, 3.3],
        description="ratio (range_pair_float); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on erasing parameters, value or sweep."""
        self._check_range_pair("scale", minimum=0.0)
        self._check_range_pair("ratio", minimum=0.0)
        self._check_p()
        return self


class ImageCompressionTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``image_compression`` transformation."""

    type: Literal["image_compression"] = "image_compression"
    compression_type: Literal["jpeg", "webp"] = Field(
        default="jpeg",
        description="compression_type (enum); a SweepConfig expands it.",
    )
    quality_range: list[int] | int | SweepConfig = Field(
        default=[99, 100],
        description="quality_range (range_pair_int); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on image_compression parameters, value or sweep."""
        self._check_range_pair("quality_range", minimum=1, maximum=100)
        self._check_p()
        return self


class DownscaleTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``downscale`` transformation."""

    type: Literal["downscale"] = "downscale"
    scale_range: list[float] | float | SweepConfig = Field(
        default=[0.25, 0.25],
        description="scale_range (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on downscale parameters, value or sweep."""
        self._check_range_pair("scale_range", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class RandomCropTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_crop`` transformation."""

    type: Literal["random_crop"] = "random_crop"
    height: int | SweepConfig = Field(
        description="height (size); a SweepConfig expands it.",
    )
    width: int | SweepConfig = Field(
        description="width (size); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_crop parameters, value or sweep."""
        self._check_parameter_bounds("height", minimum=1)
        self._check_parameter_bounds("width", minimum=1)
        self._check_p()
        return self


class PadIfNeededTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``pad_if_needed`` transformation."""

    type: Literal["pad_if_needed"] = "pad_if_needed"
    min_height: int | SweepConfig = Field(
        default=1024,
        description="min_height (size); a SweepConfig expands it.",
    )
    min_width: int | SweepConfig = Field(
        default=1024,
        description="min_width (size); a SweepConfig expands it.",
    )
    pad_height_divisor: int | SweepConfig | None = Field(
        default=None,
        description="pad_height_divisor (size); a SweepConfig expands it.",
    )
    pad_width_divisor: int | SweepConfig | None = Field(
        default=None,
        description="pad_width_divisor (size); a SweepConfig expands it.",
    )
    position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"] = Field(
        default="center",
        description="position (enum); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on pad_if_needed parameters, value or sweep."""
        self._check_parameter_bounds("min_height", minimum=1)
        self._check_parameter_bounds("min_width", minimum=1)
        self._check_parameter_bounds("pad_height_divisor", minimum=1)
        self._check_parameter_bounds("pad_width_divisor", minimum=1)
        self._check_p()
        return self


class GaussianBlurTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``gaussian_blur`` transformation."""

    type: Literal["gaussian_blur"] = "gaussian_blur"
    sigma_limit: list[float] | float | SweepConfig = Field(
        default=[0.5, 3.0],
        description="sigma_limit (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on gaussian_blur parameters, value or sweep."""
        self._check_range_pair("sigma_limit", minimum=0.0)
        self._check_p()
        return self


class RandomBrightnessContrastTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_brightness_contrast`` transformation."""

    type: Literal["random_brightness_contrast"] = "random_brightness_contrast"
    brightness_limit: list[float] | float | SweepConfig = Field(
        default=[-0.2, 0.2],
        description="brightness_limit (range_pair_float); a SweepConfig expands it.",
    )
    contrast_limit: list[float] | float | SweepConfig = Field(
        default=[-0.2, 0.2],
        description="contrast_limit (range_pair_float); a SweepConfig expands it.",
    )
    ensure_safe_range: bool = Field(
        default=False,
        description="ensure_safe_range (bool); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_brightness_contrast parameters, value or sweep."""
        self._check_range_pair("brightness_limit", minimum=-1.0, maximum=1.0)
        self._check_range_pair("contrast_limit", minimum=-1.0, maximum=1.0)
        self._check_p()
        return self


class D4TransformationConfig(AlbumentationsConfig):
    """Configuration of the ``d4`` transformation."""

    type: Literal["d4"] = "d4"

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on d4 parameters, value or sweep."""
        self._check_p()
        return self


class SquareSymmetryTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``square_symmetry`` transformation."""

    type: Literal["square_symmetry"] = "square_symmetry"

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on square_symmetry parameters, value or sweep."""
        self._check_p()
        return self


class SafeRotateTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``safe_rotate`` transformation."""

    type: Literal["safe_rotate"] = "safe_rotate"
    limit: list[float] | float | SweepConfig = Field(
        default=[-90.0, 90.0],
        description="limit (range_pair_float); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on safe_rotate parameters, value or sweep."""
        self._check_range_pair("limit", minimum=-360.0, maximum=360.0)
        self._check_p()
        return self


class AffineTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``affine`` transformation."""

    type: Literal["affine"] = "affine"
    scale: list[float] | float | SweepConfig = Field(
        default=[1.0, 1.0],
        description="scale (range_pair_float); a SweepConfig expands it.",
    )
    rotate: list[float] | float | SweepConfig = Field(
        default=0.0,
        description="rotate (range_pair_float); a SweepConfig expands it.",
    )
    shear: list[float] | float | SweepConfig = Field(
        default=[0.0, 0.0],
        description="shear (range_pair_float); a SweepConfig expands it.",
    )
    translate_percent: list[float] | float | SweepConfig | None = Field(
        default=None,
        description="translate_percent (range_pair_float); a SweepConfig expands it.",
    )
    translate_px: list[int] | int | SweepConfig | None = Field(
        default=None,
        description="translate_px (range_pair_int); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on affine parameters, value or sweep."""
        self._check_range_pair("scale", minimum=0.0)
        self._check_range_pair("rotate", minimum=-360.0, maximum=360.0)
        self._check_range_pair("shear", minimum=-360.0, maximum=360.0)
        self._check_range_pair("translate_percent", minimum=-1.0, maximum=1.0)
        self._check_range_pair("translate_px")
        self._check_p()
        return self


class PerspectiveTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``perspective`` transformation."""

    type: Literal["perspective"] = "perspective"
    scale: list[float] | float | SweepConfig = Field(
        default=[0.05, 0.1],
        description="scale (range_pair_float); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on perspective parameters, value or sweep."""
        self._check_range_pair("scale", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class GridDistortionTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``grid_distortion`` transformation."""

    type: Literal["grid_distortion"] = "grid_distortion"
    num_steps: float | int | SweepConfig = Field(
        default=5,
        description="num_steps (scalar); a SweepConfig expands it.",
    )
    distort_limit: list[float] | float | SweepConfig = Field(
        default=[-0.3, 0.3],
        description="distort_limit (range_pair_float); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on grid_distortion parameters, value or sweep."""
        self._check_parameter_bounds("num_steps", minimum=1)
        self._check_range_pair("distort_limit", minimum=-1.0, maximum=1.0)
        self._check_p()
        return self


class OpticalDistortionTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``optical_distortion`` transformation."""

    type: Literal["optical_distortion"] = "optical_distortion"
    distort_limit: list[float] | float | SweepConfig = Field(
        default=[-0.05, 0.05],
        description="distort_limit (range_pair_float); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on optical_distortion parameters, value or sweep."""
        self._check_range_pair("distort_limit", minimum=-1.0, maximum=1.0)
        self._check_p()
        return self


class PiecewiseAffineTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``piecewise_affine`` transformation."""

    type: Literal["piecewise_affine"] = "piecewise_affine"
    scale: list[float] | float | SweepConfig = Field(
        default=[0.03, 0.05],
        description="scale (range_pair_float); a SweepConfig expands it.",
    )
    nb_rows: list[int] | int | SweepConfig = Field(
        default=[4, 4],
        description="nb_rows (range_pair_int); a SweepConfig expands it.",
    )
    nb_cols: list[int] | int | SweepConfig = Field(
        default=[4, 4],
        description="nb_cols (range_pair_int); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on piecewise_affine parameters, value or sweep."""
        self._check_range_pair("scale", minimum=0.0)
        self._check_range_pair("nb_rows", minimum=1)
        self._check_range_pair("nb_cols", minimum=1)
        self._check_p()
        return self


class ThinPlateSplineTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``thin_plate_spline`` transformation."""

    type: Literal["thin_plate_spline"] = "thin_plate_spline"
    scale_range: list[float] | float | SweepConfig = Field(
        default=[0.2, 0.4],
        description="scale_range (range_pair_float); a SweepConfig expands it.",
    )
    num_control_points: float | int | SweepConfig = Field(
        default=4,
        description="num_control_points (scalar); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on thin_plate_spline parameters, value or sweep."""
        self._check_range_pair("scale_range", minimum=0.0)
        self._check_parameter_bounds("num_control_points", minimum=4)
        self._check_p()
        return self


class GridElasticDeformTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``grid_elastic_deform`` transformation."""

    type: Literal["grid_elastic_deform"] = "grid_elastic_deform"
    num_grid_xy: list[int] | int | SweepConfig = Field(
        description="num_grid_xy (range_pair_int); a SweepConfig expands it.",
    )
    magnitude: float | int | SweepConfig = Field(
        description="magnitude (scalar); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on grid_elastic_deform parameters, value or sweep."""
        self._check_range_pair("num_grid_xy", minimum=1)
        self._check_parameter_bounds("magnitude", minimum=0)
        self._check_p()
        return self


class BlurTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``blur`` transformation."""

    type: Literal["blur"] = "blur"
    blur_limit: list[int] | int | SweepConfig = Field(
        default=[3, 7],
        description="blur_limit (range_pair_int); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on blur parameters, value or sweep."""
        self._check_range_pair("blur_limit", minimum=3, maximum=255)
        self._check_p()
        return self


class AdvancedBlurTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``advanced_blur`` transformation."""

    type: Literal["advanced_blur"] = "advanced_blur"
    blur_limit: list[int] | int | SweepConfig = Field(
        default=[3, 7],
        description="blur_limit (range_pair_int); a SweepConfig expands it.",
    )
    sigma_x_limit: list[float] | float | SweepConfig = Field(
        default=[0.2, 1.0],
        description="sigma_x_limit (range_pair_float); a SweepConfig expands it.",
    )
    sigma_y_limit: list[float] | float | SweepConfig = Field(
        default=[0.2, 1.0],
        description="sigma_y_limit (range_pair_float); a SweepConfig expands it.",
    )
    rotate_limit: list[float] | float | SweepConfig = Field(
        default=[-90.0, 90.0],
        description="rotate_limit (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on advanced_blur parameters, value or sweep."""
        self._check_range_pair("blur_limit", minimum=3, maximum=255)
        self._check_range_pair("sigma_x_limit", minimum=0.0)
        self._check_range_pair("sigma_y_limit", minimum=0.0)
        self._check_range_pair("rotate_limit", minimum=-360.0, maximum=360.0)
        self._check_p()
        return self


class DefocusTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``defocus`` transformation."""

    type: Literal["defocus"] = "defocus"
    radius: list[int] | int | SweepConfig = Field(
        default=[3, 10],
        description="radius (range_pair_int); a SweepConfig expands it.",
    )
    alias_blur: list[float] | float | SweepConfig = Field(
        default=[0.1, 0.5],
        description="alias_blur (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on defocus parameters, value or sweep."""
        self._check_range_pair("radius", minimum=1)
        self._check_range_pair("alias_blur", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class ZoomBlurTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``zoom_blur`` transformation."""

    type: Literal["zoom_blur"] = "zoom_blur"
    max_factor: list[float] | float | SweepConfig = Field(
        default=[1.0, 1.31],
        description="max_factor (range_pair_float); a SweepConfig expands it.",
    )
    step_factor: list[float] | float | SweepConfig = Field(
        default=[0.01, 0.03],
        description="step_factor (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on zoom_blur parameters, value or sweep."""
        self._check_range_pair("max_factor", minimum=1.0)
        self._check_range_pair("step_factor", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class SharpenTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``sharpen`` transformation."""

    type: Literal["sharpen"] = "sharpen"
    alpha: list[float] | float | SweepConfig = Field(
        default=[0.2, 0.5],
        description="alpha (range_pair_float); a SweepConfig expands it.",
    )
    lightness: list[float] | float | SweepConfig = Field(
        default=[0.5, 1.0],
        description="lightness (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on sharpen parameters, value or sweep."""
        self._check_range_pair("alpha", minimum=0.0, maximum=1.0)
        self._check_range_pair("lightness", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class RingingOvershootTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``ringing_overshoot`` transformation."""

    type: Literal["ringing_overshoot"] = "ringing_overshoot"
    blur_limit: list[int] | int | SweepConfig = Field(
        default=[7, 15],
        description="blur_limit (range_pair_int); a SweepConfig expands it.",
    )
    cutoff: list[float] | float | SweepConfig = Field(
        default=[0.7853981633974483, 1.5707963267948966],
        description="cutoff (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on ringing_overshoot parameters, value or sweep."""
        self._check_range_pair("blur_limit", minimum=3, maximum=255)
        self._check_range_pair("cutoff")
        self._check_p()
        return self


class AdditiveNoiseTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``additive_noise`` transformation."""

    type: Literal["additive_noise"] = "additive_noise"
    noise_type: Literal["uniform", "gaussian"] = Field(
        default="uniform",
        description="noise_type (enum); a SweepConfig expands it.",
    )
    approximation: float | int | SweepConfig = Field(
        default=1.0,
        description="approximation (scalar); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on additive_noise parameters, value or sweep."""
        self._check_parameter_bounds("approximation", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class MultiplicativeNoiseTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``multiplicative_noise`` transformation."""

    type: Literal["multiplicative_noise"] = "multiplicative_noise"
    multiplier: list[float] | float | SweepConfig = Field(
        default=[0.9, 1.1],
        description="multiplier (range_pair_float); a SweepConfig expands it.",
    )
    per_channel: bool = Field(
        default=False,
        description="per_channel (bool); a SweepConfig expands it.",
    )
    elementwise: bool = Field(
        default=False,
        description="elementwise (bool); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on multiplicative_noise parameters, value or sweep."""
        self._check_range_pair("multiplier", minimum=0.0)
        self._check_p()
        return self


class ShotNoiseTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``shot_noise`` transformation."""

    type: Literal["shot_noise"] = "shot_noise"
    scale_range: list[float] | float | SweepConfig = Field(
        default=[0.1, 0.3],
        description="scale_range (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on shot_noise parameters, value or sweep."""
        self._check_range_pair("scale_range", minimum=0.0)
        self._check_p()
        return self


class IsonoiseTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``isonoise`` transformation."""

    type: Literal["isonoise"] = "isonoise"
    color_shift: list[float] | float | SweepConfig = Field(
        default=[0.01, 0.05],
        description="color_shift (range_pair_float); a SweepConfig expands it.",
    )
    intensity: list[float] | float | SweepConfig = Field(
        default=[0.1, 0.5],
        description="intensity (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on isonoise parameters, value or sweep."""
        self._check_range_pair("color_shift", minimum=0.0)
        self._check_range_pair("intensity", minimum=0.0)
        self._check_p()
        return self


class AutoContrastTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``auto_contrast`` transformation."""

    type: Literal["auto_contrast"] = "auto_contrast"
    cutoff: float | int | SweepConfig = Field(
        default=0,
        description="cutoff (scalar); a SweepConfig expands it.",
    )
    method: Literal["cdf", "pixel"] = Field(
        default="cdf",
        description="method (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on auto_contrast parameters, value or sweep."""
        self._check_parameter_bounds("cutoff", minimum=0, maximum=100)
        self._check_p()
        return self


class ClaheTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``clahe`` transformation."""

    type: Literal["clahe"] = "clahe"
    clip_limit: float | int | SweepConfig = Field(
        default=4.0,
        description="clip_limit (scalar); a SweepConfig expands it.",
    )
    tile_grid_size: list[int] | int | SweepConfig = Field(
        default=[8, 8],
        description="tile_grid_size (range_pair_int); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on clahe parameters, value or sweep."""
        self._check_parameter_bounds("clip_limit", minimum=0.0)
        self._check_range_pair("tile_grid_size", minimum=1)
        self._check_p()
        return self


class FancyPcaTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``fancy_pca`` transformation."""

    type: Literal["fancy_pca"] = "fancy_pca"
    alpha: float | int | SweepConfig = Field(
        default=0.1,
        description="alpha (scalar); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on fancy_pca parameters, value or sweep."""
        self._check_parameter_bounds("alpha", minimum=0.0)
        self._check_p()
        return self


class InvertImgTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``invert_img`` transformation."""

    type: Literal["invert_img"] = "invert_img"

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on invert_img parameters, value or sweep."""
        self._check_p()
        return self


class PosterizeTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``posterize`` transformation."""

    type: Literal["posterize"] = "posterize"
    num_bits: float | int | SweepConfig = Field(
        default=4,
        description="num_bits (scalar); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on posterize parameters, value or sweep."""
        self._check_parameter_bounds("num_bits", minimum=1, maximum=8)
        self._check_p()
        return self


class RgbShiftTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``rgb_shift`` transformation."""

    type: Literal["rgb_shift"] = "rgb_shift"
    r_shift_limit: list[int] | int | SweepConfig = Field(
        default=[-20, 20],
        description="r_shift_limit (range_pair_int); a SweepConfig expands it.",
    )
    g_shift_limit: list[int] | int | SweepConfig = Field(
        default=[-20, 20],
        description="g_shift_limit (range_pair_int); a SweepConfig expands it.",
    )
    b_shift_limit: list[int] | int | SweepConfig = Field(
        default=[-20, 20],
        description="b_shift_limit (range_pair_int); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on rgb_shift parameters, value or sweep."""
        self._check_range_pair("r_shift_limit", minimum=-255, maximum=255)
        self._check_range_pair("g_shift_limit", minimum=-255, maximum=255)
        self._check_range_pair("b_shift_limit", minimum=-255, maximum=255)
        self._check_p()
        return self


class RandomToneCurveTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_tone_curve`` transformation."""

    type: Literal["random_tone_curve"] = "random_tone_curve"
    scale: float | int | SweepConfig = Field(
        default=0.1,
        description="scale (scalar); a SweepConfig expands it.",
    )
    per_channel: bool = Field(
        default=False,
        description="per_channel (bool); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_tone_curve parameters, value or sweep."""
        self._check_parameter_bounds("scale", minimum=0.0)
        self._check_p()
        return self


class PlanckianJitterTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``planckian_jitter`` transformation."""

    type: Literal["planckian_jitter"] = "planckian_jitter"
    mode: Literal["blackbody", "cied"] = Field(
        default="blackbody",
        description="mode (enum); a SweepConfig expands it.",
    )
    sampling_method: Literal["uniform", "gaussian"] = Field(
        default="uniform",
        description="sampling_method (enum); a SweepConfig expands it.",
    )
    temperature_limit: list[int] | int | SweepConfig | None = Field(
        default=None,
        description="temperature_limit (range_pair_int); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on planckian_jitter parameters, value or sweep."""
        self._check_range_pair("temperature_limit", minimum=1000, maximum=15000)
        self._check_p()
        return self


class IlluminationTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``illumination`` transformation."""

    type: Literal["illumination"] = "illumination"
    mode: Literal["linear", "corner", "gaussian"] = Field(
        default="linear",
        description="mode (enum); a SweepConfig expands it.",
    )
    intensity_range: list[float] | float | SweepConfig = Field(
        default=[0.01, 0.2],
        description="intensity_range (range_pair_float); a SweepConfig expands it.",
    )
    effect_type: Literal["both", "brighten", "darken"] = Field(
        default="both",
        description="effect_type (enum); a SweepConfig expands it.",
    )
    angle_range: list[float] | float | SweepConfig = Field(
        default=[0.0, 360.0],
        description="angle_range (range_pair_float); a SweepConfig expands it.",
    )
    sigma_range: list[float] | float | SweepConfig = Field(
        default=[0.2, 1.0],
        description="sigma_range (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on illumination parameters, value or sweep."""
        self._check_range_pair("intensity_range", minimum=0.0, maximum=1.0)
        self._check_range_pair("angle_range", minimum=0.0, maximum=360.0)
        self._check_range_pair("sigma_range", minimum=0.0)
        self._check_p()
        return self


class PlasmaBrightnessContrastTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``plasma_brightness_contrast`` transformation."""

    type: Literal["plasma_brightness_contrast"] = "plasma_brightness_contrast"
    brightness_range: list[float] | float | SweepConfig = Field(
        default=[-0.3, 0.3],
        description="brightness_range (range_pair_float); a SweepConfig expands it.",
    )
    contrast_range: list[float] | float | SweepConfig = Field(
        default=[-0.3, 0.3],
        description="contrast_range (range_pair_float); a SweepConfig expands it.",
    )
    plasma_size: float | int | SweepConfig = Field(
        default=256,
        description="plasma_size (scalar); a SweepConfig expands it.",
    )
    roughness: float | int | SweepConfig = Field(
        default=3.0,
        description="roughness (scalar); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on plasma_brightness_contrast parameters, value or sweep."""
        self._check_range_pair("brightness_range", minimum=-1.0, maximum=1.0)
        self._check_range_pair("contrast_range", minimum=-1.0, maximum=1.0)
        self._check_parameter_bounds("plasma_size", minimum=1)
        self._check_parameter_bounds("roughness", minimum=0.0)
        self._check_p()
        return self


class PlasmaShadowTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``plasma_shadow`` transformation."""

    type: Literal["plasma_shadow"] = "plasma_shadow"
    shadow_intensity_range: list[float] | float | SweepConfig = Field(
        default=[0.3, 0.7],
        description="shadow_intensity_range (range_pair_float); a SweepConfig expands it.",
    )
    plasma_size: float | int | SweepConfig = Field(
        default=256,
        description="plasma_size (scalar); a SweepConfig expands it.",
    )
    roughness: float | int | SweepConfig = Field(
        default=3.0,
        description="roughness (scalar); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on plasma_shadow parameters, value or sweep."""
        self._check_range_pair("shadow_intensity_range", minimum=0.0, maximum=1.0)
        self._check_parameter_bounds("plasma_size", minimum=1)
        self._check_parameter_bounds("roughness", minimum=0.0)
        self._check_p()
        return self


class MorphologicalTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``morphological`` transformation."""

    type: Literal["morphological"] = "morphological"
    scale: list[int] | int | SweepConfig = Field(
        default=[2, 3],
        description="scale (range_pair_int); a SweepConfig expands it.",
    )
    operation: Literal["dilation", "erosion"] = Field(
        default="dilation",
        description="operation (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on morphological parameters, value or sweep."""
        self._check_range_pair("scale", minimum=1)
        self._check_p()
        return self


class ChannelShuffleTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``channel_shuffle`` transformation."""

    type: Literal["channel_shuffle"] = "channel_shuffle"

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on channel_shuffle parameters, value or sweep."""
        self._check_p()
        return self


class ChannelDropoutTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``channel_dropout`` transformation."""

    type: Literal["channel_dropout"] = "channel_dropout"
    channel_drop_range: list[int] | int | SweepConfig = Field(
        default=[1, 1],
        description="channel_drop_range (range_pair_int); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on channel_dropout parameters, value or sweep."""
        self._check_range_pair("channel_drop_range", minimum=0)
        self._check_p()
        return self


class ChromaticAberrationTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``chromatic_aberration`` transformation."""

    type: Literal["chromatic_aberration"] = "chromatic_aberration"
    primary_distortion_limit: list[float] | float | SweepConfig = Field(
        default=[-0.02, 0.02],
        description="primary_distortion_limit (range_pair_float); a SweepConfig expands it.",
    )
    secondary_distortion_limit: list[float] | float | SweepConfig = Field(
        default=[-0.05, 0.05],
        description="secondary_distortion_limit (range_pair_float); a SweepConfig expands it.",
    )
    mode: Literal["green_purple", "red_blue", "random"] = Field(
        default="green_purple",
        description="mode (enum); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on chromatic_aberration parameters, value or sweep."""
        self._check_range_pair("primary_distortion_limit", minimum=-1.0, maximum=1.0)
        self._check_range_pair("secondary_distortion_limit", minimum=-1.0, maximum=1.0)
        self._check_p()
        return self


class GridDropoutTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``grid_dropout`` transformation."""

    type: Literal["grid_dropout"] = "grid_dropout"
    ratio: float | int | SweepConfig = Field(
        default=0.5,
        description="ratio (scalar); a SweepConfig expands it.",
    )
    unit_size_range: list[int] | int | SweepConfig | None = Field(
        default=None,
        description="unit_size_range (range_pair_int); a SweepConfig expands it.",
    )
    holes_number_xy: list[int] | int | SweepConfig | None = Field(
        default=None,
        description="holes_number_xy (range_pair_int); a SweepConfig expands it.",
    )
    random_offset: bool = Field(
        default=True,
        description="random_offset (bool); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on grid_dropout parameters, value or sweep."""
        self._check_parameter_bounds("ratio", minimum=0.0, maximum=1.0)
        self._check_range_pair("unit_size_range", minimum=2)
        self._check_range_pair("holes_number_xy", minimum=1)
        self._check_p()
        return self


class PixelDropoutTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``pixel_dropout`` transformation."""

    type: Literal["pixel_dropout"] = "pixel_dropout"
    dropout_prob: float | int | SweepConfig = Field(
        default=0.01,
        description="dropout_prob (scalar); a SweepConfig expands it.",
    )
    per_channel: bool = Field(
        default=False,
        description="per_channel (bool); a SweepConfig expands it.",
    )
    drop_value: float | int | SweepConfig = Field(
        default=0,
        description="drop_value (scalar); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on pixel_dropout parameters, value or sweep."""
        self._check_parameter_bounds("dropout_prob", minimum=0.0, maximum=1.0)
        self._check_parameter_bounds("drop_value", minimum=0, maximum=255)
        self._check_p()
        return self


class XyMaskingTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``xy_masking`` transformation."""

    type: Literal["xy_masking"] = "xy_masking"
    num_masks_x: float | int | SweepConfig = Field(
        default=0,
        description="num_masks_x (scalar); a SweepConfig expands it.",
    )
    num_masks_y: float | int | SweepConfig = Field(
        default=0,
        description="num_masks_y (scalar); a SweepConfig expands it.",
    )
    mask_x_length: float | int | SweepConfig = Field(
        default=0,
        description="mask_x_length (scalar); a SweepConfig expands it.",
    )
    mask_y_length: float | int | SweepConfig = Field(
        default=0,
        description="mask_y_length (scalar); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on xy_masking parameters, value or sweep."""
        self._check_parameter_bounds("num_masks_x", minimum=0)
        self._check_parameter_bounds("num_masks_y", minimum=0)
        self._check_parameter_bounds("mask_x_length", minimum=0)
        self._check_parameter_bounds("mask_y_length", minimum=0)
        self._check_p()
        return self


class FrequencyMaskingTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``frequency_masking`` transformation."""

    type: Literal["frequency_masking"] = "frequency_masking"
    freq_mask_param: float | int | SweepConfig = Field(
        default=30,
        description="freq_mask_param (scalar); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on frequency_masking parameters, value or sweep."""
        self._check_parameter_bounds("freq_mask_param", minimum=0)
        self._check_p()
        return self


class RandomGridShuffleTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_grid_shuffle`` transformation."""

    type: Literal["random_grid_shuffle"] = "random_grid_shuffle"
    grid: list[int] | int | SweepConfig = Field(
        default=[3, 3],
        description="grid (range_pair_int); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_grid_shuffle parameters, value or sweep."""
        self._check_range_pair("grid", minimum=1)
        self._check_p()
        return self


class RandomFogTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_fog`` transformation."""

    type: Literal["random_fog"] = "random_fog"
    alpha_coef: float | int | SweepConfig = Field(
        default=0.08,
        description="alpha_coef (scalar); a SweepConfig expands it.",
    )
    fog_coef_range: list[float] | float | SweepConfig = Field(
        default=[0.3, 1.0],
        description="fog_coef_range (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_fog parameters, value or sweep."""
        self._check_parameter_bounds("alpha_coef", minimum=0.0, maximum=1.0)
        self._check_range_pair("fog_coef_range", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class RandomRainTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_rain`` transformation."""

    type: Literal["random_rain"] = "random_rain"
    slant_range: list[int] | int | SweepConfig = Field(
        default=[-10, 10],
        description="slant_range (range_pair_int); a SweepConfig expands it.",
    )
    drop_length: float | int | SweepConfig | None = Field(
        default=None,
        description="drop_length (scalar); a SweepConfig expands it.",
    )
    drop_width: float | int | SweepConfig = Field(
        default=1,
        description="drop_width (scalar); a SweepConfig expands it.",
    )
    blur_value: float | int | SweepConfig = Field(
        default=7,
        description="blur_value (scalar); a SweepConfig expands it.",
    )
    brightness_coefficient: float | int | SweepConfig = Field(
        default=0.7,
        description="brightness_coefficient (scalar); a SweepConfig expands it.",
    )
    rain_type: Literal["default", "drizzle", "heavy", "torrential"] = Field(
        default="default",
        description="rain_type (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_rain parameters, value or sweep."""
        self._check_range_pair("slant_range", minimum=-45, maximum=45)
        self._check_parameter_bounds("drop_length", minimum=1)
        self._check_parameter_bounds("drop_width", minimum=1)
        self._check_parameter_bounds("blur_value", minimum=0)
        self._check_parameter_bounds("brightness_coefficient", minimum=0.0)
        self._check_p()
        return self


class RandomShadowTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_shadow`` transformation."""

    type: Literal["random_shadow"] = "random_shadow"
    num_shadows_limit: list[int] | int | SweepConfig = Field(
        default=[1, 2],
        description="num_shadows_limit (range_pair_int); a SweepConfig expands it.",
    )
    shadow_dimension: float | int | SweepConfig = Field(
        default=5,
        description="shadow_dimension (scalar); a SweepConfig expands it.",
    )
    shadow_intensity_range: list[float] | float | SweepConfig = Field(
        default=[0.5, 0.5],
        description="shadow_intensity_range (range_pair_float); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_shadow parameters, value or sweep."""
        self._check_range_pair("num_shadows_limit", minimum=1)
        self._check_parameter_bounds("shadow_dimension", minimum=1)
        self._check_range_pair("shadow_intensity_range", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class RandomSnowTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_snow`` transformation."""

    type: Literal["random_snow"] = "random_snow"
    brightness_coeff: float | int | SweepConfig = Field(
        default=2.5,
        description="brightness_coeff (scalar); a SweepConfig expands it.",
    )
    snow_point_range: list[float] | float | SweepConfig = Field(
        default=[0.1, 0.3],
        description="snow_point_range (range_pair_float); a SweepConfig expands it.",
    )
    method: Literal["bleach", "texture"] = Field(
        default="bleach",
        description="method (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_snow parameters, value or sweep."""
        self._check_parameter_bounds("brightness_coeff", minimum=0.0)
        self._check_range_pair("snow_point_range", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class RandomSunFlareTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_sun_flare`` transformation."""

    type: Literal["random_sun_flare"] = "random_sun_flare"
    src_radius: float | int | SweepConfig = Field(
        default=400,
        description="src_radius (scalar); a SweepConfig expands it.",
    )
    angle_range: list[float] | float | SweepConfig = Field(
        default=[0.0, 1.0],
        description="angle_range (range_pair_float); a SweepConfig expands it.",
    )
    num_flare_circles_range: list[int] | int | SweepConfig = Field(
        default=[6, 10],
        description="num_flare_circles_range (range_pair_int); a SweepConfig expands it.",
    )
    method: Literal["overlay", "physics_based"] = Field(
        default="overlay",
        description="method (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_sun_flare parameters, value or sweep."""
        self._check_parameter_bounds("src_radius", minimum=1)
        self._check_range_pair("angle_range", minimum=0.0, maximum=1.0)
        self._check_range_pair("num_flare_circles_range", minimum=0)
        self._check_p()
        return self


class SpatterTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``spatter`` transformation."""

    type: Literal["spatter"] = "spatter"
    mean: list[float] | float | SweepConfig = Field(
        default=[0.65, 0.65],
        description="mean (range_pair_float); a SweepConfig expands it.",
    )
    std: list[float] | float | SweepConfig = Field(
        default=[0.3, 0.3],
        description="std (range_pair_float); a SweepConfig expands it.",
    )
    gauss_sigma: list[float] | float | SweepConfig = Field(
        default=[2.0, 2.0],
        description="gauss_sigma (range_pair_float); a SweepConfig expands it.",
    )
    cutout_threshold: list[float] | float | SweepConfig = Field(
        default=[0.68, 0.68],
        description="cutout_threshold (range_pair_float); a SweepConfig expands it.",
    )
    intensity: list[float] | float | SweepConfig = Field(
        default=[0.6, 0.6],
        description="intensity (range_pair_float); a SweepConfig expands it.",
    )
    mode: Literal["rain", "mud"] = Field(
        default="rain",
        description="mode (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on spatter parameters, value or sweep."""
        self._check_range_pair("mean", minimum=0.0, maximum=1.0)
        self._check_range_pair("std", minimum=0.0, maximum=1.0)
        self._check_range_pair("gauss_sigma", minimum=0.0)
        self._check_range_pair("cutout_threshold", minimum=0.0, maximum=1.0)
        self._check_range_pair("intensity", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class CenterCropTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``center_crop`` transformation."""

    type: Literal["center_crop"] = "center_crop"
    height: int | SweepConfig = Field(
        description="height (size); a SweepConfig expands it.",
    )
    width: int | SweepConfig = Field(
        description="width (size); a SweepConfig expands it.",
    )
    pad_if_needed: bool = Field(
        default=False,
        description="pad_if_needed (bool); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on center_crop parameters, value or sweep."""
        self._check_parameter_bounds("height", minimum=1)
        self._check_parameter_bounds("width", minimum=1)
        self._check_p()
        return self


class CropTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``crop`` transformation."""

    type: Literal["crop"] = "crop"
    x_min: int | SweepConfig = Field(
        default=0,
        description="x_min (size); a SweepConfig expands it.",
    )
    y_min: int | SweepConfig = Field(
        default=0,
        description="y_min (size); a SweepConfig expands it.",
    )
    x_max: int | SweepConfig = Field(
        default=1024,
        description="x_max (size); a SweepConfig expands it.",
    )
    y_max: int | SweepConfig = Field(
        default=1024,
        description="y_max (size); a SweepConfig expands it.",
    )
    pad_if_needed: bool = Field(
        default=False,
        description="pad_if_needed (bool); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on crop parameters, value or sweep."""
        self._check_parameter_bounds("x_min", minimum=0)
        self._check_parameter_bounds("y_min", minimum=0)
        self._check_parameter_bounds("x_max", minimum=1)
        self._check_parameter_bounds("y_max", minimum=1)
        self._check_p()
        return self


class RandomCropFromBordersTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_crop_from_borders`` transformation."""

    type: Literal["random_crop_from_borders"] = "random_crop_from_borders"
    crop_left: float | int | SweepConfig = Field(
        default=0.1,
        description="crop_left (scalar); a SweepConfig expands it.",
    )
    crop_right: float | int | SweepConfig = Field(
        default=0.1,
        description="crop_right (scalar); a SweepConfig expands it.",
    )
    crop_top: float | int | SweepConfig = Field(
        default=0.1,
        description="crop_top (scalar); a SweepConfig expands it.",
    )
    crop_bottom: float | int | SweepConfig = Field(
        default=0.1,
        description="crop_bottom (scalar); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_crop_from_borders parameters, value or sweep."""
        self._check_parameter_bounds("crop_left", minimum=0.0, maximum=1.0)
        self._check_parameter_bounds("crop_right", minimum=0.0, maximum=1.0)
        self._check_parameter_bounds("crop_top", minimum=0.0, maximum=1.0)
        self._check_parameter_bounds("crop_bottom", minimum=0.0, maximum=1.0)
        self._check_p()
        return self


class RandomScaleTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_scale`` transformation."""

    type: Literal["random_scale"] = "random_scale"
    scale_limit: list[float] | float | SweepConfig = Field(
        default=[-0.1, 0.1],
        description="scale_limit (range_pair_float); a SweepConfig expands it.",
    )
    area_for_downscale: Literal["none", "image", "image_mask"] = Field(
        default="none",
        description="area_for_downscale (enum); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_scale parameters, value or sweep."""
        self._check_range_pair("scale_limit", minimum=-1.0)
        self._check_p()
        return self


class RandomResizedCropTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_resized_crop`` transformation."""

    type: Literal["random_resized_crop"] = "random_resized_crop"
    size: int | SweepConfig = Field(
        description="size (size); a SweepConfig expands it.",
    )
    scale: list[float] | float | SweepConfig = Field(
        default=[0.08, 1.0],
        description="scale (range_pair_float); a SweepConfig expands it.",
    )
    ratio: list[float] | float | SweepConfig = Field(
        default=[0.75, 1.3333333333333333],
        description="ratio (range_pair_float); a SweepConfig expands it.",
    )
    area_for_downscale: Literal["none", "image", "image_mask"] = Field(
        default="none",
        description="area_for_downscale (enum); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_resized_crop parameters, value or sweep."""
        self._check_parameter_bounds("size", minimum=1)
        self._check_range_pair("scale", minimum=0.0, maximum=1.0)
        self._check_range_pair("ratio", minimum=0.0)
        self._check_p()
        return self


class RandomSizedCropTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``random_sized_crop`` transformation."""

    type: Literal["random_sized_crop"] = "random_sized_crop"
    min_max_height: list[int] | int | SweepConfig = Field(
        description="min_max_height (range_pair_int); a SweepConfig expands it.",
    )
    size: int | SweepConfig = Field(
        description="size (size); a SweepConfig expands it.",
    )
    w2h_ratio: float | int | SweepConfig = Field(
        default=1.0,
        description="w2h_ratio (scalar); a SweepConfig expands it.",
    )
    area_for_downscale: Literal["none", "image", "image_mask"] = Field(
        default="none",
        description="area_for_downscale (enum); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on random_sized_crop parameters, value or sweep."""
        self._check_range_pair("min_max_height", minimum=1)
        self._check_parameter_bounds("size", minimum=1)
        self._check_parameter_bounds("w2h_ratio", minimum=0.0)
        self._check_p()
        return self


class ResizeTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``resize`` transformation."""

    type: Literal["resize"] = "resize"
    height: int | SweepConfig = Field(
        description="height (size); a SweepConfig expands it.",
    )
    width: int | SweepConfig = Field(
        description="width (size); a SweepConfig expands it.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic", "area", "lanczos"] = Field(
        default="bilinear",
        description="interpolation (enum); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on resize parameters, value or sweep."""
        self._check_parameter_bounds("height", minimum=1)
        self._check_parameter_bounds("width", minimum=1)
        self._check_p()
        return self


class PadTransformationConfig(AlbumentationsConfig):
    """Configuration of the ``pad`` transformation."""

    type: Literal["pad"] = "pad"
    padding: float | int | SweepConfig = Field(
        default=0,
        description="padding (scalar); a SweepConfig expands it.",
    )
    border_mode: Literal["constant", "replicate", "reflect", "wrap", "reflect_101"] = Field(
        default="constant",
        description="border_mode (enum); a SweepConfig expands it.",
    )
    fill: int = Field(
        default=0,
        description="fill (fill); a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the algorithm bounds on pad parameters, value or sweep."""
        self._check_parameter_bounds("padding", minimum=0)
        self._check_p()
        return self
