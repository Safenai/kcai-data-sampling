# kcai-data-sampling-art

Adversarial attacks from the [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
(ART) as kcai data-sampling transformations.

## What it ships today

Gradient attacks, against your target model's `grad` (classifiers and
detectors alike), all deterministic:

- `art_fgsm` — ART's `FastGradientMethod`: one L-infinity step of budget
  `epsilon` (normalized `[0, 1]` pixel units).
- `art_pgd` — ART's `ProjectedGradientDescent`: `max_iter` steps of
  `step_fraction * epsilon`, projected into the L-infinity or L2 ball
  (`norm`). No random start: ART only allows it on classifiers.
- `art_mim` — ART's `MomentumIterativeMethod`: `art_pgd` with momentum
  (`decay`).

Black-box attacks, that only query your target model's `predict` (class
scores), so they also attack inference-only models. Stochastic: they need a
`seed`.

- `art_square` — ART's `SquareAttack` (L-infinity): a random search over
  square-shaped updates.
- `art_simba` — ART's `SimBA`, pixel mode (L-infinity): one random pixel
  value at a time moves by `±epsilon` when it lowers the current class's
  probability. Needs probabilities, not logits. Weak on large images that the
  model downsizes.
- `art_hopskipjump` — ART's `HopSkipJump`: decision-based, uses only the top
  class. **No budget**: it returns the closest misclassified image it finds.
  Holds up to `max_eval` noisy copies of the image in memory.

## How it works

ART attacks take an *estimator*; kcai gives a *target model*. Two bridges, in
`estimator.py`:

- `TargetEstimator` (gradient attacks): `loss_gradient` forwards to your
  `grad`; the labels ART threads through are unused. Any target that works
  with `fgsm` works with `art_fgsm`, `art_pgd` and `art_mim` unchanged.
- `TargetClassifier` (black-box attacks): `predict` forwards to your
  `predict`, in batches of at most the attack's batch size, checked to
  return `(B, n_classes)` scores.

Both keep kcai's layout (`(B, H, W, C)`, channels last) and units (`[0, 1]`,
`clip_values = (0, 1)`). See `docs/target-models.md` for writing a target.

## Install

Not published yet, and not part of the umbrella's extras: the package is
available from a checkout of the repository.

```bash
uv sync --package kcai-data-sampling-art         # with uv
pip install -e packages/kcai-data-sampling-art   # with pip
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
    - name: art_pgd
      type: art_pgd
      target_model: yolo
      epsilon: 0.03
      norm: inf                     # or 2
      max_iter: 10
      step_fraction: 0.25
    - name: art_hopskipjump
      type: art_hopskipjump
      target_model: my_classifier   # needs predict
      max_iter: 10
      max_eval: 100
      seed: 0
```

See `examples/config/walkthrough-adversarial-art.yaml`.
