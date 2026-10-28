# kcai-data-sampling-albumentation

Albumentations adapters for the data-sampling pipeline. The package provides a
unary image transformation catalogue (aliased under
`kcai_data_sampling_albumentations.transforms`) built from a curated allowlist of
Albumentations classes, with automatic schema extraction, dataclass-based
configs, clipping/backpressure, and integration with the `Output` contract.

## Quick start

### Install

```bash
pip install kcai-data-sampling-albumentation
```

### Run the walkthrough

From the repository root, use the walkthrough configuration (examples live in the
repo):

```bash
python -m kcai_data_sampling.cli run examples/configs/walkthrough_albumentations.yaml
```

Or via the published index flow (if using the packaging test helpers), ensure
the package is installed in the environment. The walkthrough reads the sample
data under `examples/data/woodscape_sample/` and writes outputs under
`examples/out/walkthrough_albumentations/` with a recipe and ledger.

## What this package does

- **Curated catalogue**: A fixed allowlist of Albumentations transforms exposed
  as unary image transforms with consistent naming and config schemas.
- **Schema-driven configs**: Dataclasses and JSON-schema generation for
  transform parameters; unknown or disallowed parameters are rejected.
- **Geometry-safe handling**: Transforms that change geometry are tagged
  (`resize_back`, `crop_back`) so the pipeline can track size semantics; the
  contract remains unary image-only (no batch-wise structural assumptions).
- **Robust execution**: Clipping to valid ranges, dtype handling, and
  integration with the pipeline's `Output` (deterministic ids, recipe
  emission).
- **Discoverable**: Transforms are registered under
  `kcai_data_sampling_albumentations.transforms` and surfaced by the plugin
  system.

## Scope
- The filters are based on the main usage scenarios
- In this version, only classification is integrated. detection and segmentation (bbox, mask, ... supports) will be added in a future release.

## Application probability (`p`)

Every catalogue entry inherits a shared `p` parameter: the probability the
transform is actually applied to a given sample. Default is `1.0` (always
apply) — deliberately not upstream's `0.5`, so a sampling entry never silently
emits identity outputs mislabeled as transformed. `p` can be fixed or swept
like any other parameter.

```yaml
transformations:
  - name: blur
    type: gaussian_blur
    seed: 11
    p: 0.5                      # apply to half the samples
    sigma_limit: [0.5, 3.0]
  - name: contrast
    type: random_brightness_contrast
    seed: 42
    p:                          # sweep probability across variants
      range: [0.25, 1.0]
      samples: 3
      mode: even
    brightness_limit: [0.1, 0.5]
    contrast_limit: [-0.2, 0.2]
```

`p` is part of the resolved parameters, so it feeds the sample `id` and the
replay contract. All entries also accept `p`; it is omitted from the
per-entry parameter lists below for brevity.

## Catalogue

The catalogue is the allowlist (see *Scope* above). Each family is represented:
geometry/dihedral, blur, noise, color/tone, occlusion/dropout, weather and
compression.

The package ships the catalogue, not the examples: the walkthrough config and
the sample data live in the repository (see *Run the walkthrough* above).

| Type | Upstream class | Family | Exposed parameters | Geometry |
|---|---|---|---|---|
| `vertical_flip` | `VerticalFlip` | dihedral | — | — |
| `rotate` | `Rotate` | geometric | `limit`, `interpolation`, `border_mode`, `fill` | — |
| `shift_scale_rotate` | `ShiftScaleRotate` | geometric | `shift_limit`, `scale_limit`, `rotate_limit`, `interpolation`, `border_mode`, `fill` | — |
| `elastic_transform` | `ElasticTransform` | geometric | `alpha`, `sigma`, `interpolation`, `border_mode`, `fill` | — |
| `median_blur` | `MedianBlur` | blur | `blur_limit` | — |
| `motion_blur` | `MotionBlur` | blur | `blur_limit`, `allow_shifted`, `angle_range`, `direction_range` | — |
| `unsharp_mask` | `UnsharpMask` | blur | `blur_limit`, `sigma_limit`, `alpha`, `threshold` | — |
| `gauss_noise` | `GaussNoise` | noise | `std_range`, `mean_range`, `per_channel`, `noise_scale_factor` | — |
| `salt_and_pepper` | `SaltAndPepper` | noise | `amount`, `salt_vs_pepper` | — |
| `color_jitter` | `ColorJitter` | color | `brightness`, `contrast`, `saturation`, `hue` | — |
| `hue_saturation_value` | `HueSaturationValue` | color | `hue_shift_limit`, `sat_shift_limit`, `val_shift_limit` | — |
| `random_gamma` | `RandomGamma` | color | `gamma_limit` | — |
| `solarize` | `Solarize` | color | `threshold_range` | — |
| `equalize` | `Equalize` | color | `mode`, `by_channels` | — |
| `coarse_dropout` | `CoarseDropout` | dropout | `num_holes_range`, `hole_height_range`, `hole_width_range`, `fill` | — |
| `erasing` | `Erasing` | dropout | `scale`, `ratio`, `fill` | — |
| `image_compression` | `ImageCompression` | compression | `compression_type`, `quality_range` | — |
| `downscale` | `Downscale` | compression | `scale_range` | — |
| `random_crop` | `RandomCrop` | geometric | `height`, `width`, `border_mode`, `fill` | `resize_back` |
| `pad_if_needed` | `PadIfNeeded` | geometric | `min_height`, `min_width`, `position`, `border_mode`, `fill` | `crop_back` |
| `gaussian_blur` | `GaussianBlur` | blur | `sigma_limit` | — |
| `random_brightness_contrast` | `RandomBrightnessContrast` | color | `brightness_limit`, `contrast_limit`, `ensure_safe_range` | — |
| `random_rain` | `RandomRain` | weather | `slant_range`, `drop_length` (omit for auto), `drop_width`, `blur_value`, `brightness_coefficient`, `rain_type` | — |
| `d4` | `D4` | dihedral | — | — |
| `square_symmetry` | `SquareSymmetry` | dihedral | — | — |
| `safe_rotate` | `SafeRotate` | geometric | `limit`, `interpolation`, `border_mode`, `fill` | — |
| `affine` | `Affine` | geometric | `scale`, `rotate`, `shear`, `interpolation`, `border_mode`, `fill` | — |
| `perspective` | `Perspective` | geometric | `scale`, `interpolation`, `border_mode`, `fill` | — |
| `grid_distortion` | `GridDistortion` | geometric | `num_steps`, `distort_limit`, `interpolation`, `border_mode`, `fill` | — |
| `optical_distortion` | `OpticalDistortion` | geometric | `distort_limit`, `interpolation`, `border_mode`, `fill` | — |
| `piecewise_affine` | `PiecewiseAffine` | geometric | `scale`, `nb_rows`, `nb_cols`, `interpolation`, `border_mode`, `fill` | — |
| `thin_plate_spline` | `ThinPlateSpline` | geometric | `scale_range`, `num_control_points`, `interpolation`, `border_mode`, `fill` | — |
| `grid_elastic_deform` | `GridElasticDeform` | geometric | `num_grid_xy`, `magnitude`, `interpolation` | — |
| `blur` | `Blur` | blur | `blur_limit` | — |
| `advanced_blur` | `AdvancedBlur` | blur | `blur_limit`, `sigma_x_limit`, `sigma_y_limit`, `rotate_limit` | — |
| `defocus` | `Defocus` | blur | `radius`, `alias_blur` | — |
| `zoom_blur` | `ZoomBlur` | blur | `max_factor`, `step_factor` | — |
| `sharpen` | `Sharpen` | blur | `alpha`, `lightness` | — |
| `ringing_overshoot` | `RingingOvershoot` | blur | `blur_limit`, `cutoff` | — |
| `additive_noise` | `AdditiveNoise` | noise | `noise_type`, `approximation` | — |
| `multiplicative_noise` | `MultiplicativeNoise` | noise | `multiplier`, `per_channel`, `elementwise` | — |
| `shot_noise` | `ShotNoise` | noise | `scale_range` | — |
| `isonoise` | `ISONoise` | noise | `color_shift`, `intensity` | — |
| `auto_contrast` | `AutoContrast` | color | `cutoff`, `method` | — |
| `clahe` | `CLAHE` | color | `clip_limit`, `tile_grid_size` | — |
| `fancy_pca` | `FancyPCA` | color | `alpha` | — |
| `invert_img` | `InvertImg` | color | — | — |
| `posterize` | `Posterize` | color | `num_bits` | — |
| `rgb_shift` | `RGBShift` | color | `r_shift_limit`, `g_shift_limit`, `b_shift_limit` | — |
| `random_tone_curve` | `RandomToneCurve` | color | `scale`, `per_channel` | — |
| `planckian_jitter` | `PlanckianJitter` | color | `mode`, `sampling_method` | — |
| `illumination` | `Illumination` | color | `mode`, `intensity_range`, `effect_type`, `angle_range`, `sigma_range` | — |
| `plasma_brightness_contrast` | `PlasmaBrightnessContrast` | color | `brightness_range`, `contrast_range`, `plasma_size`, `roughness` | — |
