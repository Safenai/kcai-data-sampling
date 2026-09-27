# kcai-data-sampling-core

Core contracts and machinery for data sampling: the transformation interface
(`api/`), configuration models (`models/`), and registries/runner/matching utils
(`utils/`).

## Loading a user model without a plugin

A `models:` entry can reference the user's own code instead of a registered
plugin package. Three routes, from the least ceremony to shipping your own
package:

1. **The file route** — `type: python` with `path:` (or `module:`), pointers
   to the user's own source, resolved at `JobConfig` load:

   ```yaml
   models:
     my_yolo:
       type: python
       path: adapters/yolo_target.py   # absolute path, or relative to the working directory
       export: YoloTarget              # optional; defaults to the models: key name
       weights: yolov8n.pt             # optional; passed to the adapter constructor
   ```

2. **The `register_model` route** — in-process registration makes a plain
   `type: my_yolo` entry valid: `register_model("my_yolo", cls)` from a kernel
   or `setup.py` hook.
3. **The plugin route** — a package declaring a `kcai_data_sampling.models`
   entry point. An entry-point plugin keeps ownership of its name and is never
   masked by the `register_model` route.

- `type: python` **executes the referenced file or module by design** — that is
  the bring-your-own-code feature. The file is user-supplied and named from the
  user's own config; nothing is downloaded or sandboxed.
- `export` is a dotted attribute path to a class, a factory callable, or an
  instance. Unset, the module attribute named like the `models:` key is used;
  else the sole model-shaped symbol (a non-empty `name` plus a `grad`/`inpaint`
  method); else `JobConfig` validation fails, listing the candidates.
- `path:` resolves absolute-first, then relative to the working directory;
  `module:` (an importable dotted path) is the spelling for a package that
  needs relative imports.

A worked target model is shipped alongside the examples:
`examples/config/walkthrough-adversarial.yaml` drives
`examples/adapters/yolo_target.py` (a YOLOv8 target whose `grad` you replace
with your own loss). A `type: python` reference is refused at `JobConfig` load
when its file is missing, its `export` names nothing, or no model-shaped
symbol is exposed.