# kcai-data-sampling-art

Adversarial attacks from the [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
(ART) as kcai data-sampling transformations.

## What it ships today

- `art_fgsm` — ART's `FastGradientMethod` (L-infinity, one step of budget
  `epsilon` in normalized `[0, 1]` pixel units), against your target model.

## How it works

ART attacks take an *estimator*; kcai gives a *target model* (`name` + `grad`).
`TargetEstimator` is the adapter: `loss_gradient` forwards to your `grad`, the
labels ART threads through are unused, and `clip_values` is `(0, 1)`. Any
target that works with `fgsm` works with `art_fgsm` unchanged.

## Install

```bash
pip install "kcai-data-sampling[art]"
```

ART's core needs numpy, scipy and scikit-learn only — no deep-learning
framework. The gradient comes from *your* model.

## Use

```yaml
operations:
  transformations:
    - name: art_fgsm
      type: art_fgsm
      target_model: yolo
      epsilon: 0.05
```

See `examples/config/walkthrough-adversarial-art.yaml`.
