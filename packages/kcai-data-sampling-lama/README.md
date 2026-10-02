# kcai-data-sampling-lama

The **generative** slot of kcai data-sampling: transformations where a model
*produces* the output. Here the model is a **tool** — it fills in content that
was not in the source sample, so the family's defining property is that the
output is genuinely new rather than a rearrangement of the input.

This package implements inpainting with
[LaMa](https://github.com/advimman/lama) (Suvorov et al., WACV 2022) as a tool
model, and exposes it through two entry-point groups: `inpaint` under
`kcai_data_sampling.transformations`, and `lama_inpaint` under
`kcai_data_sampling.models`.

A tool model declares `model_role = "tool"`, which is what makes the framework
treat it as generative. The map is deterministic — no randomness is drawn, so
the recorded seed is `None` — and not reversible, because what was in the filled
region is gone.

## What it ships today

- `inpaint` — one method: erase a rectangular region and let the model refill it.
- `lama_inpaint` — the LaMa tool model behind it.

One method, as an example of the slot rather than a catalogue.

## Install

```bash
pip install "kcai-data-sampling[lama]"
```

**This is the package that brings torch.** It is the only one of the
transformation packages that depends on a deep-learning framework, so an install
that does not ask for it stays torch-free. `torch` is imported lazily — importing
this package does not itself require torch — and the checkpoint is a weights
file fetched at run time, not a dependency.

The released *big-lama* TorchScript checkpoint (~200 MB) is downloaded into the
weight cache on first use, from the upstream release. The cache is
`$KCAI_WEIGHTS_DIR` when that is set; otherwise `.cache/` beside the working
directory — inside a checkout that resolves to the repository's own `.cache/`.
Checkpoints are never committed.

## Example

The `models` and `operations` sections of a job configuration:

```yaml
models:
  lama:
    type: lama_inpaint
    weights: big-lama.pt
    params: { margin: 256 }
operations:
  transformations:
    - name: inpaint
      type: inpaint
      tool_model: lama
      top: 350
      left: 700
      height: 300
      width: 500
```

The transformation names its model with `tool_model: lama`, referring to the
key under `models:`. `top`/`left`/`height`/`width` describe the rectangle to
erase; the window must fit inside the frame, which is only knowable at run time,
so a window that would leave it is refused rather than silently wrapped.
`margin` is how many pixels of surrounding context are cropped around the mask
for the network.

Only the channels the network consumes are sent to it — three, RGB — so the
alpha plane of an RGBA batch is passed through untouched.

## Bring your own model

Any object reachable through a `kcai_data_sampling.models` entry point is a
model, exactly like a transformation. To use a custom inference routine, ship a
small importable package that registers your adapter under that entry-point group
and lets it declare its own dependencies (`torch`, your inference library) in its
own metadata. They never belong to the core contracts or to the job package.
A classic-sketch adapter to paste into your package:

```python
class MyModel:
    channels = 3  # RGB only: the adapter receives exactly those planes

    def __init__(self, weights: str, **params):
        self.name = weights  # the ledger records this string
        self._load(weights, **params)

    def inpaint(self, xs, masks):
        """(B, H, W, C) uint8 image batch in, (B, H, W, C) uint8 out."""
        ...  # fill every masked pixel; leave the rest unchanged


def plugin() -> MyModel:
    return MyModel
```

and point `[project.entry-points."kcai_data_sampling.models"]` at it (e.g.
`mymodel = "my_package.models:plugin"`). A tool model is recognised by a
non-empty `name` plus an `inpaint` method, and conformance is checked against
the `tool` role the transformation declares.

Known-architecture custom weights need no code at all: keep `type: lama_inpaint`
and set `weights` to your own checkpoint file.