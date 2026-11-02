"""The declarative allowlist: one spec row per integrated upstream transform.

Data only — no albumentations import, no logic beyond dataclasses. Adding a
transform means adding a row here and re-running ``_generate.py``; the
generated configs, transformation classes and entry-point block are committed,
and a freshness test keeps them in sync with this table. The parameter kinds
and their schema/bounds mapping live in ``config_base.py``.
"""

from dataclasses import dataclass, field
from typing import Any

#: Upstream parameter kinds → generated schema fields (see config_base.py).
ParamKind = str


@dataclass(frozen=True)
class Param:
    """One exposed upstream parameter.

    Attributes:
        kind: The parameter kind (``range_pair_float``, ``range_pair_int``,
            ``scalar``, ``size``, ``enum``, ``bool``, ``fill``).
        default: The upstream default (what our schema pins when the user
            omits the parameter). ``...`` (Ellipsis) marks an upstream-required
            parameter, which the schema leaves required too.
        bounds: Inclusive ``(minimum, maximum)`` for the value (or each
            element of a pair); ``None`` leaves a side open.
        choices: For ``enum``: friendly name → upstream value — a cv2 code for
            geometric knobs (``bilinear → 1``), the upstream string for string
            enums (``jpeg → "jpeg"``).
    """

    kind: ParamKind
    default: Any = None
    bounds: tuple[float | None, float | None] = (None, None)
    choices: dict[str, Any] | None = None


@dataclass(frozen=True)
class AlbuSpec:
    """One registered transformation: our snake_case name → upstream class.

    Attributes:
        source: The upstream albumentations class name.
        family: Grouping tag for docs and review (not the workspace family,
            which stays ``"procedural"``).
        params: Exposed parameters, keyed by upstream parameter name.
        geometry: ``"resize_back"`` (crop family), ``"crop_back"`` (pad
            family) or ``None`` (shape-preserving).
        pinned: Upstream-only kwargs fixed by this table (shape-affecting
            knobs like ``keep_size=True``), not exposed in the config.
    """

    source: str
    family: str
    params: dict[str, Param] = field(default_factory=dict)
    geometry: str | None = None
    pinned: dict[str, Any] = field(default_factory=dict)


#: Friendly names for the cv2 interpolation and border codes the geometric
#: transforms expose. Both appear upstream as raw ints; the schema keeps the
#: name and the engine maps it back (``enum_maps`` on the generated class).
_INTERPOLATION = {"nearest": 0, "bilinear": 1, "bicubic": 2, "area": 3, "lanczos": 4}
_BORDER_MODE = {"constant": 0, "replicate": 1, "reflect": 2, "wrap": 3, "reflect_101": 4}

TRANSFORM_SPECS: dict[str, AlbuSpec] = {
    # --- dihedral ---
    "vertical_flip": AlbuSpec(source="VerticalFlip", family="dihedral"),
    # --- geometric: shape-preserving, Tier-A ---
    "rotate": AlbuSpec(
        source="Rotate",
        family="geometric",
        params={
            "limit": Param(kind="range_pair_float", default=(-90, 90), bounds=(-360.0, 360.0)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "shift_scale_rotate": AlbuSpec(
        source="ShiftScaleRotate",
        family="geometric",
        params={
            "shift_limit": Param(kind="range_pair_float", default=(-0.0625, 0.0625), bounds=(-1.0, 1.0)),
            "shift_limit_x": Param(kind="range_pair_float", default=None, bounds=(-1.0, 1.0)),
            "shift_limit_y": Param(kind="range_pair_float", default=None, bounds=(-1.0, 1.0)),
            "scale_limit": Param(kind="range_pair_float", default=(-0.1, 0.1), bounds=(-1.0, None)),
            "rotate_limit": Param(kind="range_pair_float", default=(-45, 45), bounds=(-360.0, 360.0)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "elastic_transform": AlbuSpec(
        source="ElasticTransform",
        family="geometric",
        params={
            "alpha": Param(kind="scalar", default=1, bounds=(0.0, None)),
            "sigma": Param(kind="scalar", default=50, bounds=(1.0, None)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    # --- blur ---
    "median_blur": AlbuSpec(
        source="MedianBlur",
        family="blur",
        params={
            "blur_limit": Param(kind="range_pair_int", default=(3, 7), bounds=(3, 255)),
        },
    ),
    "motion_blur": AlbuSpec(
        source="MotionBlur",
        family="blur",
        params={
            "blur_limit": Param(kind="range_pair_int", default=(3, 7), bounds=(3, 255)),
            "allow_shifted": Param(kind="bool", default=True),
            "angle_range": Param(kind="range_pair_float", default=(0, 360), bounds=(0.0, 360.0)),
            "direction_range": Param(kind="range_pair_float", default=(-1.0, 1.0), bounds=(-1.0, 1.0)),
        },
    ),
    "unsharp_mask": AlbuSpec(
        source="UnsharpMask",
        family="blur",
        params={
            "blur_limit": Param(kind="range_pair_int", default=(3, 7), bounds=(3, 255)),
            "sigma_limit": Param(kind="scalar", default=0.0, bounds=(0.0, None)),
            "alpha": Param(kind="range_pair_float", default=(0.2, 0.5), bounds=(0.0, 1.0)),
            "threshold": Param(kind="scalar", default=10, bounds=(0, 255)),
        },
    ),
    # --- noise ---
    "gauss_noise": AlbuSpec(
        source="GaussNoise",
        family="noise",
        params={
            "std_range": Param(kind="range_pair_float", default=(0.2, 0.44), bounds=(0.0, 1.0)),
            "mean_range": Param(kind="range_pair_float", default=(0.0, 0.0), bounds=(-1.0, 1.0)),
            "per_channel": Param(kind="bool", default=True),
            "noise_scale_factor": Param(kind="scalar", default=1, bounds=(0.0, 1.0)),
        },
    ),
    "salt_and_pepper": AlbuSpec(
        source="SaltAndPepper",
        family="noise",
        params={
            "amount": Param(kind="range_pair_float", default=(0.01, 0.06), bounds=(0.0, 1.0)),
            "salt_vs_pepper": Param(kind="range_pair_float", default=(0.4, 0.6), bounds=(0.0, 1.0)),
        },
    ),
    # --- color / tone ---
    "color_jitter": AlbuSpec(
        source="ColorJitter",
        family="color",
        params={
            "brightness": Param(kind="range_pair_float", default=(0.8, 1.2), bounds=(0.0, None)),
            "contrast": Param(kind="range_pair_float", default=(0.8, 1.2), bounds=(0.0, None)),
            "saturation": Param(kind="range_pair_float", default=(0.8, 1.2), bounds=(0.0, None)),
            "hue": Param(kind="range_pair_float", default=(-0.5, 0.5), bounds=(-0.5, 0.5)),
        },
    ),
    "hue_saturation_value": AlbuSpec(
        source="HueSaturationValue",
        family="color",
        params={
            "hue_shift_limit": Param(kind="range_pair_int", default=(-20, 20), bounds=(-180, 180)),
            "sat_shift_limit": Param(kind="range_pair_int", default=(-30, 30), bounds=(-255, 255)),
            "val_shift_limit": Param(kind="range_pair_int", default=(-20, 20), bounds=(-255, 255)),
        },
    ),
    "random_gamma": AlbuSpec(
        source="RandomGamma",
        family="color",
        params={
            "gamma_limit": Param(kind="range_pair_int", default=(80, 120), bounds=(1, None)),
        },
    ),
    "solarize": AlbuSpec(
        source="Solarize",
        family="color",
        params={
            "threshold_range": Param(kind="range_pair_float", default=(0.5, 0.5), bounds=(0.0, 1.0)),
        },
    ),
    "equalize": AlbuSpec(
        source="Equalize",
        family="color",
        params={
            "mode": Param(kind="enum", default="cv", choices={"cv": "cv", "pil": "pil"}),
            "by_channels": Param(kind="bool", default=True),
        },
    ),
    # --- occlusion / dropout ---
    "coarse_dropout": AlbuSpec(
        source="CoarseDropout",
        family="dropout",
        params={
            "num_holes_range": Param(kind="range_pair_int", default=(1, 2), bounds=(1, None)),
            "hole_height_range": Param(kind="range_pair_float", default=(0.1, 0.2), bounds=(0.0, None)),
            "hole_width_range": Param(kind="range_pair_float", default=(0.1, 0.2), bounds=(0.0, None)),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "erasing": AlbuSpec(
        source="Erasing",
        family="dropout",
        params={
            "scale": Param(kind="range_pair_float", default=(0.02, 0.33), bounds=(0.0, None)),
            "ratio": Param(kind="range_pair_float", default=(0.3, 3.3), bounds=(0.0, None)),
            "fill": Param(kind="fill", default=0),
        },
    ),
    # --- compression / resolution ---
    "image_compression": AlbuSpec(
        source="ImageCompression",
        family="compression",
        params={
            "compression_type": Param(kind="enum", default="jpeg", choices={"jpeg": "jpeg", "webp": "webp"}),
            "quality_range": Param(kind="range_pair_int", default=(99, 100), bounds=(1, 100)),
        },
    ),
    "downscale": AlbuSpec(
        source="Downscale",
        family="compression",
        params={
            "scale_range": Param(kind="range_pair_float", default=(0.25, 0.25), bounds=(0.0, 1.0)),
        },
    ),
    # --- geometric: shape-changing, Tier-B restores the input shape ---
    "random_crop": AlbuSpec(
        source="RandomCrop",
        family="geometric",
        geometry="resize_back",
        params={
            "height": Param(kind="size", default=..., bounds=(1, None)),
            "width": Param(kind="size", default=..., bounds=(1, None)),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "pad_if_needed": AlbuSpec(
        source="PadIfNeeded",
        family="geometric",
        geometry="crop_back",
        params={
            "min_height": Param(kind="size", default=1024, bounds=(1, None)),
            "min_width": Param(kind="size", default=1024, bounds=(1, None)),
            "pad_height_divisor": Param(kind="size", default=None, bounds=(1, None)),
            "pad_width_divisor": Param(kind="size", default=None, bounds=(1, None)),
            "position": Param(
                kind="enum",
                default="center",
                choices={
                    "center": "center",
                    "top_left": "top_left",
                    "top_right": "top_right",
                    "bottom_left": "bottom_left",
                    "bottom_right": "bottom_right",
                    "random": "random",
                },
            ),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "gaussian_blur": AlbuSpec(
        source="GaussianBlur",
        family="blur",
        params={
            "sigma_limit": Param(kind="range_pair_float", default=(0.5, 3.0), bounds=(0.0, None)),
        },
    ),
    "random_brightness_contrast": AlbuSpec(
        source="RandomBrightnessContrast",
        family="color",
        params={
            "brightness_limit": Param(kind="range_pair_float", default=(-0.2, 0.2), bounds=(-1.0, 1.0)),
            "contrast_limit": Param(kind="range_pair_float", default=(-0.2, 0.2), bounds=(-1.0, 1.0)),
            "ensure_safe_range": Param(kind="bool", default=False),
        },
    ),
    # --- dihedral / symmetry (shape-preserving, Tier A) ---
    "d4": AlbuSpec(source="D4", family="dihedral"),
    "square_symmetry": AlbuSpec(source="SquareSymmetry", family="dihedral"),
    # --- geometric: shape-preserving, Tier-A ---
    "safe_rotate": AlbuSpec(
        source="SafeRotate",
        family="geometric",
        params={
            "limit": Param(kind="range_pair_float", default=(-90, 90), bounds=(-360.0, 360.0)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "affine": AlbuSpec(
        source="Affine",
        family="geometric",
        # fit_output stays upstream's default (False), which keeps (H, W);
        # pinning it here guards against an upstream default flip.
        pinned={"fit_output": False},
        params={
            "scale": Param(kind="range_pair_float", default=(1.0, 1.0), bounds=(0.0, None)),
            "rotate": Param(kind="range_pair_float", default=0.0, bounds=(-360.0, 360.0)),
            "shear": Param(kind="range_pair_float", default=(0.0, 0.0), bounds=(-360.0, 360.0)),
            "translate_percent": Param(kind="range_pair_float", default=None, bounds=(-1.0, 1.0)),
            "translate_px": Param(kind="range_pair_int", default=None, bounds=(None, None)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "perspective": AlbuSpec(
        source="Perspective",
        family="geometric",
        # keep_size keeps (H, W) by resizing the warped quad back; fit_output
        # would instead grow the canvas to contain it (shape contract breaks).
        pinned={"keep_size": True, "fit_output": False},
        params={
            "scale": Param(kind="range_pair_float", default=(0.05, 0.1), bounds=(0.0, 1.0)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "grid_distortion": AlbuSpec(
        source="GridDistortion",
        family="geometric",
        params={
            "num_steps": Param(kind="scalar", default=5, bounds=(1, None)),
            "distort_limit": Param(kind="range_pair_float", default=(-0.3, 0.3), bounds=(-1.0, 1.0)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "optical_distortion": AlbuSpec(
        source="OpticalDistortion",
        family="geometric",
        params={
            "distort_limit": Param(kind="range_pair_float", default=(-0.05, 0.05), bounds=(-1.0, 1.0)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "piecewise_affine": AlbuSpec(
        source="PiecewiseAffine",
        family="geometric",
        params={
            "scale": Param(kind="range_pair_float", default=(0.03, 0.05), bounds=(0.0, None)),
            "nb_rows": Param(kind="range_pair_int", default=(4, 4), bounds=(1, None)),
            "nb_cols": Param(kind="range_pair_int", default=(4, 4), bounds=(1, None)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "thin_plate_spline": AlbuSpec(
        source="ThinPlateSpline",
        family="geometric",
        params={
            "scale_range": Param(kind="range_pair_float", default=(0.2, 0.4), bounds=(0.0, None)),
            "num_control_points": Param(kind="scalar", default=4, bounds=(4, None)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "grid_elastic_deform": AlbuSpec(
        source="GridElasticDeform",
        family="geometric",
        params={
            "num_grid_xy": Param(kind="range_pair_int", default=..., bounds=(1, None)),
            "magnitude": Param(kind="scalar", default=..., bounds=(0, None)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
        },
    ),
    # --- blur (shape-preserving, Tier A) ---
    "blur": AlbuSpec(
        source="Blur",
        family="blur",
        params={
            "blur_limit": Param(kind="range_pair_int", default=(3, 7), bounds=(3, 255)),
        },
    ),
    "advanced_blur": AlbuSpec(
        source="AdvancedBlur",
        family="blur",
        params={
            "blur_limit": Param(kind="range_pair_int", default=(3, 7), bounds=(3, 255)),
            "sigma_x_limit": Param(kind="range_pair_float", default=(0.2, 1.0), bounds=(0.0, None)),
            "sigma_y_limit": Param(kind="range_pair_float", default=(0.2, 1.0), bounds=(0.0, None)),
            "rotate_limit": Param(kind="range_pair_float", default=(-90, 90), bounds=(-360.0, 360.0)),
        },
    ),
    "defocus": AlbuSpec(
        source="Defocus",
        family="blur",
        params={
            "radius": Param(kind="range_pair_int", default=(3, 10), bounds=(1, None)),
            "alias_blur": Param(kind="range_pair_float", default=(0.1, 0.5), bounds=(0.0, 1.0)),
        },
    ),
    "zoom_blur": AlbuSpec(
        source="ZoomBlur",
        family="blur",
        params={
            "max_factor": Param(kind="range_pair_float", default=(1, 1.31), bounds=(1.0, None)),
            "step_factor": Param(kind="range_pair_float", default=(0.01, 0.03), bounds=(0.0, 1.0)),
        },
    ),
    "sharpen": AlbuSpec(
        source="Sharpen",
        family="blur",
        params={
            "alpha": Param(kind="range_pair_float", default=(0.2, 0.5), bounds=(0.0, 1.0)),
            "lightness": Param(kind="range_pair_float", default=(0.5, 1.0), bounds=(0.0, 1.0)),
        },
    ),
    "ringing_overshoot": AlbuSpec(
        source="RingingOvershoot",
        family="blur",
        params={
            "blur_limit": Param(kind="range_pair_int", default=(7, 15), bounds=(3, 255)),
            "cutoff": Param(kind="range_pair_float", default=(0.7853981633974483, 1.5707963267948966)),
        },
    ),
    # --- noise (shape-preserving, Tier A) ---
    "additive_noise": AlbuSpec(
        source="AdditiveNoise",
        family="noise",
        params={
            "noise_type": Param(kind="enum", default="uniform", choices={"uniform": "uniform", "gaussian": "gaussian"}),
            "approximation": Param(kind="scalar", default=1.0, bounds=(0.0, 1.0)),
        },
    ),
    "multiplicative_noise": AlbuSpec(
        source="MultiplicativeNoise",
        family="noise",
        params={
            "multiplier": Param(kind="range_pair_float", default=(0.9, 1.1), bounds=(0.0, None)),
            "per_channel": Param(kind="bool", default=False),
            "elementwise": Param(kind="bool", default=False),
        },
    ),
    "shot_noise": AlbuSpec(
        source="ShotNoise",
        family="noise",
        params={
            "scale_range": Param(kind="range_pair_float", default=(0.1, 0.3), bounds=(0.0, None)),
        },
    ),
    "isonoise": AlbuSpec(
        source="ISONoise",
        family="noise",
        params={
            "color_shift": Param(kind="range_pair_float", default=(0.01, 0.05), bounds=(0.0, None)),
            "intensity": Param(kind="range_pair_float", default=(0.1, 0.5), bounds=(0.0, None)),
        },
    ),
    # --- color / tone (shape-preserving, Tier A) ---
    "auto_contrast": AlbuSpec(
        source="AutoContrast",
        family="color",
        params={
            "cutoff": Param(kind="scalar", default=0, bounds=(0, 100)),
            "method": Param(kind="enum", default="cdf", choices={"cdf": "cdf", "pixel": "pixel"}),
        },
    ),
    "clahe": AlbuSpec(
        source="CLAHE",
        family="color",
        params={
            "clip_limit": Param(kind="scalar", default=4.0, bounds=(0.0, None)),
            "tile_grid_size": Param(kind="range_pair_int", default=(8, 8), bounds=(1, None)),
        },
    ),
    "fancy_pca": AlbuSpec(
        source="FancyPCA",
        family="color",
        params={
            "alpha": Param(kind="scalar", default=0.1, bounds=(0.0, None)),
        },
    ),
    "invert_img": AlbuSpec(source="InvertImg", family="color"),
    "posterize": AlbuSpec(
        source="Posterize",
        family="color",
        params={
            "num_bits": Param(kind="scalar", default=4, bounds=(1, 8)),
        },
    ),
    "rgb_shift": AlbuSpec(
        source="RGBShift",
        family="color",
        params={
            "r_shift_limit": Param(kind="range_pair_int", default=(-20, 20), bounds=(-255, 255)),
            "g_shift_limit": Param(kind="range_pair_int", default=(-20, 20), bounds=(-255, 255)),
            "b_shift_limit": Param(kind="range_pair_int", default=(-20, 20), bounds=(-255, 255)),
        },
    ),
    "random_tone_curve": AlbuSpec(
        source="RandomToneCurve",
        family="color",
        params={
            "scale": Param(kind="scalar", default=0.1, bounds=(0.0, None)),
            "per_channel": Param(kind="bool", default=False),
        },
    ),
    "planckian_jitter": AlbuSpec(
        source="PlanckianJitter",
        family="color",
        params={
            "mode": Param(
                kind="enum",
                default="blackbody",
                choices={"blackbody": "blackbody", "cied": "cied"},
            ),
            "sampling_method": Param(
                kind="enum", default="uniform", choices={"uniform": "uniform", "gaussian": "gaussian"}
            ),
            "temperature_limit": Param(kind="range_pair_int", default=None, bounds=(1000, 15000)),
        },
    ),
    "illumination": AlbuSpec(
        source="Illumination",
        family="color",
        params={
            "mode": Param(
                kind="enum",
                default="linear",
                choices={"linear": "linear", "corner": "corner", "gaussian": "gaussian"},
            ),
            "intensity_range": Param(kind="range_pair_float", default=(0.01, 0.2), bounds=(0.0, 1.0)),
            "effect_type": Param(
                kind="enum",
                default="both",
                choices={"both": "both", "brighten": "brighten", "darken": "darken"},
            ),
            "angle_range": Param(kind="range_pair_float", default=(0, 360), bounds=(0.0, 360.0)),
            "sigma_range": Param(kind="range_pair_float", default=(0.2, 1.0), bounds=(0.0, None)),
        },
    ),
    "plasma_brightness_contrast": AlbuSpec(
        source="PlasmaBrightnessContrast",
        family="color",
        params={
            "brightness_range": Param(kind="range_pair_float", default=(-0.3, 0.3), bounds=(-1.0, 1.0)),
            "contrast_range": Param(kind="range_pair_float", default=(-0.3, 0.3), bounds=(-1.0, 1.0)),
            "plasma_size": Param(kind="scalar", default=256, bounds=(1, None)),
            "roughness": Param(kind="scalar", default=3.0, bounds=(0.0, None)),
        },
    ),
    "plasma_shadow": AlbuSpec(
        source="PlasmaShadow",
        family="color",
        params={
            "shadow_intensity_range": Param(kind="range_pair_float", default=(0.3, 0.7), bounds=(0.0, 1.0)),
            "plasma_size": Param(kind="scalar", default=256, bounds=(1, None)),
            "roughness": Param(kind="scalar", default=3.0, bounds=(0.0, None)),
        },
    ),
    "morphological": AlbuSpec(
        source="Morphological",
        family="color",
        params={
            "scale": Param(kind="range_pair_int", default=(2, 3), bounds=(1, None)),
            "operation": Param(
                kind="enum",
                default="dilation",
                choices={"dilation": "dilation", "erosion": "erosion"},
            ),
        },
    ),
    "channel_shuffle": AlbuSpec(source="ChannelShuffle", family="color"),
    "channel_dropout": AlbuSpec(
        source="ChannelDropout",
        family="color",
        params={
            "channel_drop_range": Param(kind="range_pair_int", default=(1, 1), bounds=(0, None)),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "chromatic_aberration": AlbuSpec(
        source="ChromaticAberration",
        family="color",
        params={
            "primary_distortion_limit": Param(kind="range_pair_float", default=(-0.02, 0.02), bounds=(-1.0, 1.0)),
            "secondary_distortion_limit": Param(kind="range_pair_float", default=(-0.05, 0.05), bounds=(-1.0, 1.0)),
            "mode": Param(
                kind="enum",
                default="green_purple",
                choices={"green_purple": "green_purple", "red_blue": "red_blue", "random": "random"},
            ),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
        },
    ),
    # --- occlusion / dropout (shape-preserving, Tier A) ---
    "grid_dropout": AlbuSpec(
        source="GridDropout",
        family="dropout",
        params={
            "ratio": Param(kind="scalar", default=0.5, bounds=(0.0, 1.0)),
            "unit_size_range": Param(kind="range_pair_int", default=None, bounds=(2, None)),
            "holes_number_xy": Param(kind="range_pair_int", default=None, bounds=(1, None)),
            "random_offset": Param(kind="bool", default=True),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "pixel_dropout": AlbuSpec(
        source="PixelDropout",
        family="dropout",
        params={
            "dropout_prob": Param(kind="scalar", default=0.01, bounds=(0.0, 1.0)),
            "per_channel": Param(kind="bool", default=False),
            "drop_value": Param(kind="scalar", default=0, bounds=(0, 255)),
        },
    ),
    "xy_masking": AlbuSpec(
        source="XYMasking",
        family="dropout",
        params={
            "num_masks_x": Param(kind="scalar", default=0, bounds=(0, None)),
            "num_masks_y": Param(kind="scalar", default=0, bounds=(0, None)),
            "mask_x_length": Param(kind="scalar", default=0, bounds=(0, None)),
            "mask_y_length": Param(kind="scalar", default=0, bounds=(0, None)),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "frequency_masking": AlbuSpec(
        source="FrequencyMasking",
        family="dropout",
        params={
            "freq_mask_param": Param(kind="scalar", default=30, bounds=(0, None)),
        },
    ),
    "random_grid_shuffle": AlbuSpec(
        source="RandomGridShuffle",
        family="dropout",
        params={
            "grid": Param(kind="range_pair_int", default=(3, 3), bounds=(1, None)),
        },
    ),
    # --- weather (shape-preserving, Tier A) ---
    "random_fog": AlbuSpec(
        source="RandomFog",
        family="weather",
        params={
            "alpha_coef": Param(kind="scalar", default=0.08, bounds=(0.0, 1.0)),
            "fog_coef_range": Param(kind="range_pair_float", default=(0.3, 1), bounds=(0.0, 1.0)),
        },
    ),
    "random_rain": AlbuSpec(
        source="RandomRain",
        family="weather",
        params={
            "slant_range": Param(kind="range_pair_int", default=(-10, 10), bounds=(-45, 45)),
            "drop_length": Param(kind="scalar", default=None, bounds=(1, None)),
            "drop_width": Param(kind="scalar", default=1, bounds=(1, None)),
            "blur_value": Param(kind="scalar", default=7, bounds=(0, None)),
            "brightness_coefficient": Param(kind="scalar", default=0.7, bounds=(0.0, None)),
            "rain_type": Param(
                kind="enum",
                default="default",
                choices={"default": "default", "drizzle": "drizzle", "heavy": "heavy", "torrential": "torrential"},
            ),
        },
    ),
    "random_shadow": AlbuSpec(
        source="RandomShadow",
        family="weather",
        params={
            "num_shadows_limit": Param(kind="range_pair_int", default=(1, 2), bounds=(1, None)),
            "shadow_dimension": Param(kind="scalar", default=5, bounds=(1, None)),
            "shadow_intensity_range": Param(kind="range_pair_float", default=(0.5, 0.5), bounds=(0.0, 1.0)),
        },
    ),
    "random_snow": AlbuSpec(
        source="RandomSnow",
        family="weather",
        params={
            "brightness_coeff": Param(kind="scalar", default=2.5, bounds=(0.0, None)),
            "snow_point_range": Param(kind="range_pair_float", default=(0.1, 0.3), bounds=(0.0, 1.0)),
            "method": Param(
                kind="enum",
                default="bleach",
                choices={"bleach": "bleach", "texture": "texture"},
            ),
        },
    ),
    "random_sun_flare": AlbuSpec(
        source="RandomSunFlare",
        family="weather",
        params={
            "src_radius": Param(kind="scalar", default=400, bounds=(1, None)),
            "angle_range": Param(kind="range_pair_float", default=(0, 1), bounds=(0.0, 1.0)),
            "num_flare_circles_range": Param(kind="range_pair_int", default=(6, 10), bounds=(0, None)),
            "method": Param(
                kind="enum", default="overlay", choices={"overlay": "overlay", "physics_based": "physics_based"}
            ),
        },
    ),
    "spatter": AlbuSpec(
        source="Spatter",
        family="weather",
        params={
            "mean": Param(kind="range_pair_float", default=(0.65, 0.65), bounds=(0.0, 1.0)),
            "std": Param(kind="range_pair_float", default=(0.3, 0.3), bounds=(0.0, 1.0)),
            "gauss_sigma": Param(kind="range_pair_float", default=(2, 2), bounds=(0.0, None)),
            "cutout_threshold": Param(kind="range_pair_float", default=(0.68, 0.68), bounds=(0.0, 1.0)),
            "intensity": Param(kind="range_pair_float", default=(0.6, 0.6), bounds=(0.0, 1.0)),
            "mode": Param(kind="enum", default="rain", choices={"rain": "rain", "mud": "mud"}),
        },
    ),
    # --- geometric: shape-changing, Tier-B restores the input shape ---
    "center_crop": AlbuSpec(
        source="CenterCrop",
        family="geometric",
        geometry="resize_back",
        params={
            "height": Param(kind="size", default=..., bounds=(1, None)),
            "width": Param(kind="size", default=..., bounds=(1, None)),
            "pad_if_needed": Param(kind="bool", default=False),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "crop": AlbuSpec(
        source="Crop",
        family="geometric",
        geometry="resize_back",
        params={
            "x_min": Param(kind="size", default=0, bounds=(0, None)),
            "y_min": Param(kind="size", default=0, bounds=(0, None)),
            "x_max": Param(kind="size", default=1024, bounds=(1, None)),
            "y_max": Param(kind="size", default=1024, bounds=(1, None)),
            "pad_if_needed": Param(kind="bool", default=False),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
    "random_crop_from_borders": AlbuSpec(
        source="RandomCropFromBorders",
        family="geometric",
        geometry="resize_back",
        params={
            "crop_left": Param(kind="scalar", default=0.1, bounds=(0.0, 1.0)),
            "crop_right": Param(kind="scalar", default=0.1, bounds=(0.0, 1.0)),
            "crop_top": Param(kind="scalar", default=0.1, bounds=(0.0, 1.0)),
            "crop_bottom": Param(kind="scalar", default=0.1, bounds=(0.0, 1.0)),
        },
    ),
    "random_scale": AlbuSpec(
        source="RandomScale",
        family="geometric",
        geometry="resize_back",
        params={
            "scale_limit": Param(kind="range_pair_float", default=(-0.1, 0.1), bounds=(-1.0, None)),
            "area_for_downscale": Param(
                kind="enum",
                default="none",
                choices={"none": None, "image": "image", "image_mask": "image_mask"},
            ),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
        },
    ),
    "random_resized_crop": AlbuSpec(
        source="RandomResizedCrop",
        family="geometric",
        geometry="resize_back",
        params={
            "size": Param(kind="size", default=..., bounds=(1, None)),
            "scale": Param(kind="range_pair_float", default=(0.08, 1.0), bounds=(0.0, 1.0)),
            "ratio": Param(kind="range_pair_float", default=(0.75, 1.3333333333333333), bounds=(0.0, None)),
            "area_for_downscale": Param(
                kind="enum",
                default="none",
                choices={"none": None, "image": "image", "image_mask": "image_mask"},
            ),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
        },
    ),
    "random_sized_crop": AlbuSpec(
        source="RandomSizedCrop",
        family="geometric",
        geometry="resize_back",
        params={
            "min_max_height": Param(kind="range_pair_int", default=..., bounds=(1, None)),
            "size": Param(kind="size", default=..., bounds=(1, None)),
            "w2h_ratio": Param(kind="scalar", default=1.0, bounds=(0.0, None)),
            "area_for_downscale": Param(
                kind="enum",
                default="none",
                choices={"none": None, "image": "image", "image_mask": "image_mask"},
            ),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
        },
    ),
    "resize": AlbuSpec(
        source="Resize",
        family="geometric",
        geometry="resize_back",
        params={
            "height": Param(kind="size", default=..., bounds=(1, None)),
            "width": Param(kind="size", default=..., bounds=(1, None)),
            "interpolation": Param(kind="enum", default="bilinear", choices=_INTERPOLATION),
        },
    ),
    "pad": AlbuSpec(
        source="Pad",
        family="geometric",
        geometry="crop_back",
        params={
            "padding": Param(kind="scalar", default=0, bounds=(0, None)),
            "border_mode": Param(kind="enum", default="constant", choices=_BORDER_MODE),
            "fill": Param(kind="fill", default=0),
        },
    ),
}
