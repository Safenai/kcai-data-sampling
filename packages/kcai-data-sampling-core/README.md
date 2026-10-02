# kcai-data-sampling-core

The contracts that everything else in kcai data-sampling implements.

A **transformation** here is one fully specified operation:

    T : x ↦ x'

one input sample to one output sample, in the same space. "Fully specified"
means the algorithm, its resolved parameters, a seed where the algorithm draws
randomness, and a model where one is involved — four things that together
determine the output, so a result can be reproduced from its own row.

This package holds no algorithms and no IO. It is deliberately generic: nothing
here assumes images, and nothing here touches the disk. It provides the
interface in terms of a sample's axes, dtype and value range; the pydantic
configuration schema; and the plugin registry that discovers everything else.

## What it ships today

- `api/` — the transformation interface (arity, families, model roles), the
  batch and output contracts, and the matching used to align a transformation
  against a selection's value range.
- `models/` — the pydantic configuration schema, including sweep parameters.
- `utils/` — the entry-point plugin registry, the in-memory transformation
  runner, and model loading.

No algorithms. The methods live in their own packages — see the sibling
distributions for the procedural, generative and adversarial examples.

## Install

```bash
pip install kcai-data-sampling-core
```

This is the base library. It is also what the umbrella's extras pull in, so most
readers arrive here via `pip install "kcai-data-sampling[...]"` rather than
installing it directly.

---

## Adding your own method

The rest of this README is a guide: how to decide which family a method belongs
to, how to add one to an existing package, how to ship a new package, how to
write a model adapter, and one complete worked example.

### 1. Which family does my method belong to?

You do not choose the family. It is decided by **the model's role** in producing
the output:

| Model role | Family | Meaning |
| --- | --- | --- |
| no model | **procedural** | computed from `x` alone |
| **tool** | **generative** | the model produces the content that becomes the output |
| **target** | **adversarial** | the model's response enters the computation, and the result is graded against it |

The role says where the model plugs in; the family says what the map does. The
two are not independent — the family is derived from the role, so declaring a
role is all you do. An algorithm may declare `family` explicitly when it needs
to; otherwise the role decides.

What you must **not** declare is the regime — perturbation, augmentation or
corruption. That is read off the input/output pair *afterwards*, by comparing
`x` and `x'`. It is not a property you assert about your own algorithm.

Transformations listed together are applied **independently** to the same source
sample, not chained.

### 2. Adding a method to an existing package

Three files, in the shape the repository already uses for its own methods. Take
`horizontal_flip` as the reference — its math is one line.

**1. `transformations/<name>.py` — the math alone.**

A pure function mapping `(B, *sample)` to `(B, *sample)`, importing nothing from
the framework. Keeping the math separate is what lets you test and reuse it
without instantiating anything.

```python
def horizontal_flip(xs: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(xs[:, :, ::-1, :])
```

**2. `configs.py` — the parameter schema, next to the algorithm.**

Subclass the core `TransformationConfig` and pin `type` to the algorithm's
`Literal`. That pin is how the registry-resolved validator in `JobConfig` picks
the right schema by `type`; unknown fields are refused everywhere. A parameter
that can be swept takes a `SweepConfig`. Enforce the algorithm's ranges in a
model validator — the check works on both a plain value and a sweep interval, so
one call covers both:

```python
class CropResizeTransformationConfig(TransformationConfig):
    type: Literal["crop_resize"] = "crop_resize"
    fraction: float | SweepConfig = Field(description="Fraction (0, 1] of the image kept.")

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        self._check_parameter_bounds("fraction", minimum=0, maximum=1, exclusive_min=True)
        return self
```

Keep each algorithm's schema beside that algorithm, in the algorithm's own
package. Dependencies go the same way — with the code that needs them, never in
the core.

**3. `api/transformations/<name>.py` — the contract.**

Subclass `UnaryTransformation`, declare the algorithm name, point `Config` at the
schema, set the role flags, and implement only `apply`:

```python
class HorizontalFlip(UnaryTransformation):
    algorithm = "horizontal_flip"
    Config = HorizontalFlipTransformationConfig
    reversible = True

    @override
    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None = None) -> np.ndarray:
        del rngs
        return horizontal_flip(xs)
```

The flags are declarations, and the base enforces every one of them:

- **`clips = True`** — your output may leave the selection's value range, and the
  base clips it back. Leave it unset and an output outside the range is
  *refused*, with the offending range in the message. Set it whenever your
  operation can overshoot.
- **`model_role`** — `"tool"`, `"target"`, or unset for no model. The base
  requires that exactly the model slot named by the role is filled: a tool role
  with a target model, or neither, is an error at construction. This is also
  what derives your family.
- **`model_methods`** — the method the role needs: `("grad",)` for a target,
  `("inpaint",)` for a tool. Conformance is checked against the declared role.
- **`reversible`** — is `x` determined by `x'`? A structural claim, recorded on
  the output row. `horizontal_flip` is its own inverse, so yes; `fgsm` throws
  information away, so no.
- **`stochastic = True`** — your algorithm draws randomness. A stochastic
  algorithm **requires** a seed, and omitting it fails at construction; a
  deterministic one records `None`. When you are stochastic, `apply` receives
  one generator per output row via `rngs`, seeded from your seed, the parent ids
  and the resolved parameters — so a given sample draws the same value regardless
  of batch size or where it lands in the run. Use it rather than a global
  generator, and you get reproducibility for free.

`apply` must return **one output per input in the same shape**. If your map
changes the sample space, the base refuses it — because `δ = x' - x` has to stay
defined for the output row to mean anything.

**Then the entry point**, in the package's `pyproject.toml`:

```toml
[project.entry-points."kcai_data_sampling.transformations"]
horizontal_flip = "kcai_data_sampling_images.api.transformations.horizontal_flip:HorizontalFlip"
```

### 3. Shipping a new package instead

If your methods do not belong to an existing family package, ship your own
distribution. Nothing in the core changes.

Declare an entry point under one of four groups, and the registry discovers it
on first use:

- `kcai_data_sampling.transformations`
- `kcai_data_sampling.models`
- `kcai_data_sampling.dataloaders`
- `kcai_data_sampling.outputwriter`

Two rules follow from the design and are worth stating outright. First, **each
algorithm carries its own config schema and its own dependencies** — inside its
own package. Second, **`kcai-data-sampling-core` never grows an algorithm or an
IO plugin.** If your code needs a library, that library is a dependency of your
package, not of the core.

### 4. Writing a model adapter

An adapter is recognised by shape: a non-empty `name`, plus the method its role
needs — `grad` for a target, `inpaint` for a tool. `name` is yours to choose and
is what your config refers to.

Three routes, from least ceremony to owning your own name:

1. **`type: python` with `path:` or `module:`** — point at your own source from
   your config. Nothing is declared, installed or registered.
2. **`register_model("my_yolo", cls)`** — for something already in the process;
   makes a plain `type: my_yolo` entry valid.
3. **An entry-point plugin** under `kcai_data_sampling.models` — keeps ownership
   of its name, and is never masked by `register_model`.

The details of the first route are below, in
[Loading a user model without a plugin](#loading-a-user-model-without-a-plugin).

### 5. A complete minimal method

A `brighten` method, end to end — multiply a uint8 image batch by a factor,
sweepable across factors. Deterministic, clips, not reversible.

```python
# transformations/brighten.py
import numpy as np


def brighten(xs: np.ndarray, factor: float) -> np.ndarray:
    """Scale a uint8 batch by `factor`, in the batch's own dtype range.

    The scaling happens in float and is clipped back into `[0, 1]` before the
    cast — casting an out-of-range float straight back to uint8 would wrap
    around (400 -> 144) instead of saturating.
    """
    info = np.iinfo(xs.dtype)
    scaled = np.clip(xs.astype(np.float64) * factor / info.max, 0.0, 1.0)
    return np.rint(scaled * info.max).astype(xs.dtype)
```

```python
# configs.py
from typing import Literal, Self

from kcai_data_sampling_core.models.config import TransformationConfig
from kcai_data_sampling_core.models.sweep import SweepConfig
from pydantic import Field, model_validator


class BrightenTransformationConfig(TransformationConfig):
    """Configuration of the `brighten` transformation."""

    type: Literal["brighten"] = "brighten"
    factor: float | SweepConfig = Field(description="Multiplier applied to every value.")

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        # The upper bound is above 1 on purpose: a factor may brighten past the
        # selection's value range, which is what `clips = True` is for below.
        self._check_parameter_bounds("factor", minimum=0, maximum=4)
        return self
```

```python
# api/transformations/brighten.py
import numpy as np
from typing_extensions import override

from kcai_data_sampling_core.api.unary import UnaryTransformation

from my_package.configs import BrightenTransformationConfig
from my_package.transformations.brighten import brighten


class Brighten(UnaryTransformation):
    """Scale sample values by `factor`."""

    algorithm = "brighten"
    Config = BrightenTransformationConfig

    # `factor > 1` overshoots the selection's value range, so declare `clips`
    # and let the base clip back. (The math above clips too, but to the batch's
    # *dtype* range; this flag is what covers a selection whose own value_range
    # is narrower than that.) Scaling discards information: not reversible.
    clips = True
    reversible = False

    @override
    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None = None) -> np.ndarray:
        del rngs
        return brighten(xs, self.params["factor"])
```

```toml
# pyproject.toml
[project.entry-points."kcai_data_sampling.transformations"]
brighten = "my_package.api.transformations.brighten:Brighten"
```

No `model_role` and no `stochastic`: `brighten` uses no model and draws nothing,
so the family is derived as procedural, the recorded seed is `None`, and
`apply` ignores the generators it is handed.

That is the whole contract. The example has nothing to do with images — the same
three files work for any sample space, which is why the core does not assume
images.

---

## Loading a user model without a plugin

A `models:` entry can reference your own code instead of a registered plugin
package. Three routes, from the least ceremony to shipping your own package:

1. **The file route** — `type: python` with `path:` (or `module:`), pointers to
   your own source, resolved at `JobConfig` load:

   ```yaml
   models:
     my_yolo:
       type: python
       path: adapters/yolo_target.py   # absolute, or relative to the working directory
       export: YoloTarget              # optional; defaults to the models: key name
       weights: yolov8n.pt             # optional; passed to the adapter constructor
   ```

2. **The `register_model` route** — in-process registration makes a plain
   `type: my_yolo` entry valid: `register_model("my_yolo", cls)` from a kernel
   or a setup hook.
3. **The plugin route** — a package declaring a `kcai_data_sampling.models`
   entry point. An entry-point plugin keeps ownership of its name and is never
   masked by the `register_model` route.

- `type: python` **executes the referenced file or module by design** — that is
  the bring-your-own-code feature. The file is yours and is named from your own
  config; nothing is downloaded and nothing is sandboxed.
- `export` is a dotted attribute path to a class, a factory callable, or an
  instance. Unset, the module attribute named like the `models:` key is used;
  else the sole model-shaped symbol (a non-empty `name` plus a `grad`/`inpaint`
  method); else `JobConfig` validation fails, listing the candidates.
- `path:` resolves absolute-first, then relative to the working directory;
  `module:` (an importable dotted path) is the spelling for a package that needs
  relative imports.

A `type: python` reference is refused at `JobConfig` load when its file is
missing, its `export` names nothing, or no model-shaped symbol is exposed.