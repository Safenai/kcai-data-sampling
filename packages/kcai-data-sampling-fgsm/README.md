# kcai-data-sampling-fgsm

The **adversarial** slot of kcai data-sampling: transformations where a model's
*response* enters the computation and the result is graded against it. Here the
model is a **target** — it is not asked to produce anything, only to react. That
is what makes the family adversarial, and the declaring `model_role = "target"`
is what tells the framework to treat it as such.

This package implements the Fast Gradient Sign Method as a single budgeted step
along the sign of the target model's gradient, and exposes it through the
`kcai_data_sampling.transformations` entry-point group.

**The target model is yours.** It arrives at run time through the `models:`
section of your configuration and is never shipped by this package.

## What it ships today

- `fgsm` — one method: one L-infinity step of budget `epsilon` along the sign of
  the target's gradient.

One method, as an example of the slot rather than a catalogue.

## Install

```bash
pip install "kcai-data-sampling[fgsm]"
```

**No deep-learning framework is required.** This package depends only on the
core contracts — the gradient arrives from *your* model, so the work is yours
too, and nothing here pulls torch. A configuration driving `fgsm` may well need
torch in the environment that hosts the target model; that is a property of the
model you bring, not of this package.

## How the step works

The batch is normalized to `[0, 1]` pixel units, the target's gradient is taken
in the same units, and `epsilon * sign(grad)` is added — so only the *sign* of
the gradient participates, not its magnitude. The result is clipped back into
range and restored to the batch's dtype.

Three declarations follow from that, and the framework enforces each:

- **`clips = True`** — a signed step can push a pixel past the selection's value
  range, so the output is clipped back rather than refused.
- **`reversible = False`** — the step is lossy. The original pixels are not
  recoverable from the output, so the row records that it is not invertible.
- **Deterministic** — no randomness is drawn, so the recorded seed is `None`.
  The same input and the same `epsilon` always give the same output.

`epsilon` is the adversarial budget, strictly positive, and sweepable, so a run
can walk a whole budget curve.

## Labels do not move

An attack moves the model's *prediction*, not the truth. The ground-truth label
belongs to the source row, travels with it, and is never touched here — this
package has no notion of a label at all, only of pixels.

That is what makes an adversarial sample meaningful: if the label had moved
along with the pixels, the pair would prove nothing. A label that changed under
perturbation, or a target model whose response was ignored, would both mean the
measurement was not an attack at all.

## Example

The `models` and `operations` sections of a job configuration, with a target
model supplied as your own source:

```yaml
models:
  yolo:
    type: python
    path: adapters/yolo_target.py
    export: YoloTarget
    weights: yolov8n.pt
    params: { conf: 0.25 }
operations:
  transformations:
    - name: fgsm
      type: fgsm
      target_model: yolo
      epsilon:
        range: [0.01, 0.05]
        step: 0.01
```

The transformation names its model with `target_model: yolo`. Only `grad` is
used from it, and its output is checked for shape and finiteness before the step,
so a misbehaving adapter fails loudly rather than corrupting the batch.

A target model is recognised by a non-empty `name` plus a `grad` method, and
conformance is checked against the `target` role the transformation declares.