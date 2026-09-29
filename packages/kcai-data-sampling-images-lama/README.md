# kcai-data-sampling-images-lama

Generative inpainting with [LaMa](https://github.com/advimman/lama) (Suvorov et
al., WACV 2022) for data sampling. Adds the `inpaint` transformation
(`model_role = "tool"`) and the `lama_inpaint` model plugin, wired through the
`kcai_data_sampling.transformations` and `kcai_data_sampling.models`
entry-point groups.

`torch` lives only in this package: a default workspace install stays torch-free,
and installing this package is an explicit opt-in (`UV_INDEX= uv sync --package
kcai-data-sampling-images-lama`). The released *big-lama* TorchScript checkpoint
(~200 MB) is downloaded into the weight cache on first use — `$KCAI_WEIGHTS_DIR`
or the repository's `.cache/` — and is never committed.

Example configuration:

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

## Bring your own model

Any object reachable through a `kcai_data_sampling.models` entry point is a
model, exactly like a transformation. To use a custom inference routine, ship a
small importable package that registers your adapter under that entry-point
group and let it declare its own dependencies (`torch`, your inference
library) in its own `requirements.txt` — they never belong to `-core` or
`-job`. A classic-sketch adapter to paste into your package:

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
`mymodel = "my_package.models:plugin"`). Known-architecture custom weights need
no code at all: keep `type: lama_inpaint` and set `weights` to your own
checkpoint file.