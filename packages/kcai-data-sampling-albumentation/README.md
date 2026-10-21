# kcai-data-sampling-albumentation

The **image-robustness catalogue** of kcai data-sampling: a curated set of
[albumentations](https://albumentations.ai/) transforms, exposed as first-class
workspace transformations through the `kcai_data_sampling.transformations`
entry-point group.

Every integrated method is **procedural** — no model is involved, the transform is
pure pixel math — and every one is **stochastic**: the magnitude of the effect is
drawn per row, from the entry's seed. That combination is the point of this
package: a native procedural transform like `horizontal_flip` is deterministic,
so this is the surface where the framework's per-row seeding machinery earns its
keep.

The key decision: one shared **engine** carries all runtime behaviour, and the
methods are **generated registrations** — one table row each. The generated
`configs.py` schemas and `api/transformations/*.py` classes are committed, and a
freshness test keeps them in step with the spec table (`specs.py`).

## Install

```bash
pip install "kcai-data-sampling[albumentation]"
```

**No deep-learning framework is required.** The package depends on the core
contracts, `numpy`, and a **hard-pinned** `albumentations` — a git commit
(`66212d77a`) plus an explicit `albucore` and `opencv-python-headless`, because
the frozen permissive line has no successor (AlbumentationsX is AGPL/commercial).
It is a separate, optional extra for exactly that reason: the pin is heavy and
quarantined, so `kcai-data-sampling-images` stays tiny and the pin is one
droppable directory. Removing the integration means deleting this directory and
its umbrella extra — nothing else references it.

## Run the walkthrough

`examples/config/walkthrough-albumentation.yaml` is the living example: it runs
this package end to end on the comma10k sample. Clone the repository, fetch the
sample data, install the extra, then:

```bash
python scripts/fetch_comma10k_sample.py
kcai-data-sampling process -p examples/config/walkthrough-albumentation.yaml
```

The config shows the three things this package adds: a `[lo, hi]` per-row draw
(`gaussian_blur`), a sweep (`random_brightness_contrast` over three fixed
brightness magnitudes, with `contrast_limit` still drawn per row), and the two
Tier-B shape restores in smoke form (`random_crop` resized back, `pad_if_needed`
center-cropped back). It is also the file the end-to-end test runs, so it cannot
rot.

See
[walkthrough-albumentation.yaml](https://github.com/Safenai/kcai-data-sampling/blob/main/examples/config/walkthrough-albumentation.yaml).

## The two ways a parameter moves

This is the one real subtlety of the surface, and it is called out on every
entry:

- **A `[lo, hi]` pair is a per-row draw.** Upstream draws a fresh value in that
  range for each row, so one entry yields one transformation with a stochastic
  magnitude. Reproducible from the entry's `seed`; the ledger records the *range*
  and the seed — the drawn value is reproducible but not itself recorded (an
  accepted v1 limitation).
- **A `{range: …, samples/step: …}` mapping is a sweep.** The workspace's existing
  sweep machinery expands it into N concrete transformations, each with a
  *fixed* scalar (albumentations broadcasts a scalar to a symmetric range),
  name-suffixed and cartesian-expanded — with per-variant ledger rows and seed
  hashing, exactly as the native types behave.

The two compose: sweep to walk a curve of fixed magnitudes, or pin one config and
let each row draw its own. The syntax is unambiguous — a plain list is a range, a
mapping is a sweep.

```yaml
operations:
  transformations:
    - name: blur_moderate
      type: gaussian_blur
      seed: 42                       # REQUIRED (see below)
      sigma_limit: [0.5, 3.0]        # a pair: a per-row sigma draw
    - name: brightness_grid
      type: random_brightness_contrast
      seed: 42
      brightness_limit: {range: [-0.3, 0.3], samples: 3}   # a sweep: 3 entries
      contrast_limit: [-0.2, 0.2]                          # a per-row draw
```

## `seed` is required

Every albumentations-backed transformation is **stochastic** (`stochastic =
True`), so every entry must name a `seed`. The engine seeds one generator per row
from `[seed, sha1(row_id, params)]` and re-seeds the upstream transform before
each single-image call — so each row's output is byte-identical whatever batch it
travels in and whatever order it is processed in (the batch-invariance contract),
and the whole run replays from the ledger facts alone. This integration is the
framework's **first stochastic transformation**, so it is also the first real
user of the per-row `rngs()` seeding path.

Because skipping would otherwise emit identity outputs mislabeled as transformed,
the sampling default for `p` is **1.0** (always apply), overriding upstream's
`p = 0.5`. The stochasticity comes from the per-row draws and sweeps, not from
skipping the transform.

## The RGBA rule

Measured per family at the pinned commit:

- **Dual transforms** (geometry, flips) receive the full 4-channel row — the
  alpha plane must move with the pixels, so splitting it would be wrong.
- **Image-only transforms** (blur, noise, colour) split the alpha off, transform
  RGB, and re-attach the alpha untouched — many hard-fail on 4 channels, and the
  rest would otherwise blur or noise the alpha plane.
- 3-channel batches bypass the split; the engine adapts at runtime.

## Shape restores

The unary contract requires `(H, W)` preserved. Two tiers:

- **Tier A** (most methods) keep the shape at their pinned settings; nothing
  special happens.
- **Tier B** (the crop/pad family — `random_crop`, `pad_if_needed`) change the
  shape, so the engine restores it: crop results are resized back
  nearest-neighbor, pad results are center-cropped back. This is the same move
  the native `crop_resize` already makes, generalized; aspect distortion from
  resize-back is the accepted semantics.

## What it ships today

The allowlist is the spec table — the table below is the catalogue as it stands
(**76 registered types**; curated by judgment against the eligibility filter,
see *What is deliberately excluded* below). Each family is represented:
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
| `plasma_shadow` | `PlasmaShadow` | color | `shadow_intensity_range`, `plasma_size`, `roughness` | — |
| `morphological` | `Morphological` | color | `scale`, `operation` | — |
| `channel_shuffle` | `ChannelShuffle` | color | — | — |
| `channel_dropout` | `ChannelDropout` | color | `channel_drop_range`, `fill` | — |
| `chromatic_aberration` | `ChromaticAberration` | color | `primary_distortion_limit`, `secondary_distortion_limit`, `mode`, `interpolation` | — |
| `grid_dropout` | `GridDropout` | dropout | `ratio`, `random_offset`, `fill` | — |
| `pixel_dropout` | `PixelDropout` | dropout | `dropout_prob`, `per_channel`, `drop_value` | — |
| `xy_masking` | `XYMasking` | dropout | `num_masks_x`, `num_masks_y`, `mask_x_length`, `mask_y_length`, `fill` | — |
| `frequency_masking` | `FrequencyMasking` | dropout | `freq_mask_param` | — |
| `random_grid_shuffle` | `RandomGridShuffle` | dropout | `grid` | — |
| `random_fog` | `RandomFog` | weather | `alpha_coef`, `fog_coef_range` | — |
| `random_rain` | `RandomRain` | weather | `slant_range`, `drop_width`, `blur_value`, `brightness_coefficient`, `rain_type` | — |
| `random_shadow` | `RandomShadow` | weather | `num_shadows_limit`, `shadow_dimension`, `shadow_intensity_range` | — |
| `random_snow` | `RandomSnow` | weather | `brightness_coeff`, `snow_point_range`, `method` | — |
| `random_sun_flare` | `RandomSunFlare` | weather | `src_radius`, `angle_range`, `num_flare_circles_range`, `method` | — |
| `spatter` | `Spatter` | weather | `mean`, `std`, `gauss_sigma`, `cutout_threshold`, `intensity`, `mode` | — |
| `center_crop` | `CenterCrop` | geometric | `height`, `width`, `pad_if_needed`, `border_mode`, `fill` | `resize_back` |
| `crop` | `Crop` | geometric | `x_min`, `y_min`, `x_max`, `y_max`, `pad_if_needed`, `border_mode`, `fill` | `resize_back` |
| `random_crop_from_borders` | `RandomCropFromBorders` | geometric | `crop_left`, `crop_right`, `crop_top`, `crop_bottom` | `resize_back` |
| `random_scale` | `RandomScale` | geometric | `scale_limit`, `interpolation` | `resize_back` |
| `random_resized_crop` | `RandomResizedCrop` | geometric | `size`, `scale`, `ratio`, `interpolation` | `resize_back` |
| `random_sized_crop` | `RandomSizedCrop` | geometric | `min_max_height`, `size`, `w2h_ratio`, `interpolation` | `resize_back` |
| `resize` | `Resize` | geometric | `height`, `width`, `interpolation` | `resize_back` |
| `pad` | `Pad` | geometric | `padding`, `border_mode`, `fill` | `crop_back` |

**Enum parameters** are exposed by a friendly name and mapped to the upstream
code by the engine — interpolation (`nearest`, `bilinear`, `bicubic`, `area`,
`lanczos`), border mode (`constant`, `replicate`, `reflect`, `wrap`,
`reflect_101`), compression type (`jpeg`, `webp`), equalize mode (`cv`, `pil`)
and pad position (`center`, `top_left`, `top_right`, `bottom_left`,
`bottom_right`, `random`). String enums carry their upstream string through
unchanged: noise type (`uniform`, `gaussian`), auto-contrast method (`cdf`,
`pixel`), Planckian mode (`blackbody`, `cied`) and sampling (`uniform`,
`gaussian`), illumination mode (`linear`, `corner`, `gaussian`) and effect
(`both`, `brighten`, `darken`), morphological operation (`dilation`, `erosion`),
chromatic-aberration mode (`green_purple`, `red_blue`, `random`), rain type
(`default`, `drizzle`, `heavy`, `torrential`), snow method (`bleach`, `texture`),
sun-flare method (`overlay`, `physics_based`) and spatter mode (`rain`, `mud`).
Everything else fails fast at config load against the algorithm bounds.

## What is deliberately excluded

The allowlist is curated against a fixed eligibility filter: no
abstract bases, no 3D/volumetric or spectrogram aliases, no parameter of kind
callable/file-path/reference-image, no mask/bbox/keypoint/metadata targets (the
contract is unary image-only), no value-range or channel-count changing
transforms, nothing that swaps H×W on a non-square image, and no
serialization-hostile classes (`Lambda`, `NoOp`). An infrastructure test re-checks
the filter against the pinned upstream at test time, so the allowlist cannot
silently rot.

On top of the hard filter, the list is curated by judgment: dropped as redundant
(`HorizontalFlip` — the native type covers it), decorative (`ToSepia`, `Emboss`,
`RandomGravel`), slow/exotic (`Superpixels`, `GlassBlur`), domain-specific
(`HEStain`) or broken on RGBA (`CubicSymmetry`). Three more are excluded because
their only active knob defaults to `None` upstream, which the schema contract
cannot expose (an exposed parameter must carry upstream's own default):
`CropAndPad`, `LongestMaxSize` and `SmallestMaxSize` — each would be a permanent
no-op.

## Adding a method

Adding a transform is adding one row to `specs.py` and re-running the generator:

```bash
uv run --package kcai-data-sampling-albumentation \
  python -m kcai_data_sampling_albumentation._generate
```

Then commit the regenerated `configs.py`, the new `api/transformations/*.py`
module, the entry-point block and — if the curated set changed — this README's
table. `_generate --check` (and a test that runs it) fails if any committed
generated file drifts from the spec table.
