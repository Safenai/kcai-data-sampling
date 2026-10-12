# kcai-data-sampling-art

Adversarial attacks from the [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
(ART) as kcai data-sampling transformations.

## What it ships today

- `art_fgsm` — ART's `FastGradientMethod` (L-infinity, one step of budget
  `epsilon` in normalized `[0, 1]` pixel units), against your target model's
  `grad`.
- `art_square` — ART's `SquareAttack` (L-infinity, black-box): a random search
  that only queries your target model's `predict` (class scores), so it also
  attacks inference-only models. Stochastic: needs a `seed`.

## How it works

ART attacks take an *estimator*; kcai gives a *target model*. Two bridges, in
`estimator.py`:

- `TargetEstimator` (gradient attacks): `loss_gradient` forwards to your
  `grad`; the labels ART threads through are unused. Any target that works
  with `fgsm` works with `art_fgsm` unchanged.
- `TargetClassifier` (black-box attacks): `predict` forwards to your
  `predict`, checked to return `(B, n_classes)` scores.

Both keep kcai's layout (`(B, H, W, C)`, channels last) and units (`[0, 1]`,
`clip_values = (0, 1)`). See `docs/target-models.md` for writing a target.

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
    - name: art_square
      type: art_square
      target_model: my_classifier   # needs predict
      epsilon: 0.03
      max_iter: 100
      seed: 0
```

See `examples/config/walkthrough-adversarial-art.yaml`.
