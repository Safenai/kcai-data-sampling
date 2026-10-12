# Bring your own target model

Adversarial transformations (`fgsm`, `art_fgsm`, `art_square`) attack a **target model**: the
model under test, which is yours. kcai never loads it directly. You wrap it
in a small Python class, the *adapter*, and name that class in the job
configuration.

**The framework does not matter. What matters is what the model can give:**

- a **gradient** (`grad`) for the gradient attacks (`fgsm`, `art_fgsm`):
  PyTorch, TensorFlow/Keras and JAX models all work, through the same adapter
  shape;
- **class scores** (`predict`) for the black-box attacks (`art_square`):
  any model that can run, including inference-only ones (ONNX Runtime,
  TensorRT, OpenVINO, TFLite, a remote API). See [Limits](#limits).

## The contract

An adapter is any object with a `name` and the method(s) the transformations
you use need:

| Member | Needed by | What kcai expects |
|---|---|---|
| `name` | all | Non-empty `str`. Recorded in the ledger's `target_model` column. |
| `grad(xs)` | `fgsm`, `art_fgsm` | Gradient of **your loss** with respect to `xs`. |
| `predict(xs)` | `art_square` | Class scores `(B, n_classes)`, probabilities or logits, `n_classes >= 2`. |

Implement only what you need; implement both to use every attack. Both take
the same input, and for `grad`:

| | |
|---|---|
| Input `xs` | `numpy` float array `(B, H, W, C)` in `[0, 1]`: the `uint8` images divided by 255. `C = 4` (RGBA) for images from the parquet loader. `B` is the job's `transform_batch_size` (defaults to the loader's `load_batch_size`). |
| Output | `numpy` array of **exactly the same shape**, finite values. Give the alpha plane a zero gradient. |
| Direction | The attack moves pixels in the direction that **increases** the loss. Write the loss as "what the attacker wants more of". |
| Scale | Irrelevant for FGSM: only the sign of each gradient value is used. A zero gradient leaves the pixel unchanged. |

kcai checks the contract for you. `name` and the method a transformation
needs are checked when it is built, and the shape and finiteness of the answer
on the first call. A broken adapter fails at startup with a message naming it.

## Quick start

1. **Copy the template** `examples/adapters/template_target.py`, e.g. to
   `examples/adapters/my_target.py`. It holds only what kcai requires:

    - `name`: a readable name for the ledger;
    - `__init__`: load your model (optional if your adapter needs no setup);
    - `grad`: the gradient of your loss, with the six steps listed as comments;
    - `predict`: class scores, for the black-box attacks.

    Keep `grad`, `predict` or both, depending on the attacks you use. Until
    you fill them in, the methods raise
    `NotImplementedError`, so an unfinished adapter cannot run by mistake.
    Anything else (helper methods, extra constructor arguments) is up to you.

2. **Fill it in**, starting from the example closest to your model:
   `yolo_target.py` (PyTorch, any ultralytics detector), `keras_target.py`
   (TensorFlow, any `keras.applications` classifier) or `flax_target.py`
   (JAX, any Hugging Face Flax classifier), all in `examples/adapters/`. If
   your model is one of those, you may only need to pass its name or
   weights. See also
   [Choosing the loss](#choosing-the-loss) and [Pitfalls](#pitfalls).

3. **Test it** (optional, see [Testing your adapter](#testing-your-adapter)).

4. **Reference it in the job configuration**:

    ```yaml
    models:
      mine:                                  # the name transformations refer to
        type: python                         # your own code, not a registered plugin
        path: examples/adapters/my_target.py # or  module: my_package.adapters
        export: MyTarget                     # the class (or factory, or instance) to use
        weights: checkpoints/model.pt        # passed as MyTarget(weights=...)
        params: { device: cuda }             # passed as MyTarget(..., device="cuda") if __init__ takes it

    operations:
      transformations:
        - name: fgsm
          type: fgsm
          target_model: mine
          epsilon: 0.03                      # at most 0.03 * 255 ≈ 8 grey levels per pixel
        - name: square                       # black-box: needs predict instead of grad
          type: art_square
          target_model: mine
          epsilon: 0.03
          max_iter: 100                      # at most 100 predict calls per image
          seed: 0                            # random search: a seed is required
    ```

    The job builds the adapter as `MyTarget(weights=<weights>, **<params>)`,
    so every constructor argument can be set from the YAML. A relative `path`
    resolves against the directory you run the job from.

5. **Run** it: `kcai-data-sampling process -p my-job.yaml`.

## Testing your adapter

Optional: kcai runs the same checks when the job starts and on the first
`grad` call. Testing in Python first just gives quicker feedback than a full
job:

```python
import numpy as np
from kcai_data_sampling_core.api.roles import check_model, check_output
from kcai_data_sampling_fgsm.api.transformations.fgsm import Fgsm
from my_target import MyTarget

target = MyTarget(weights="checkpoints/model.pt")
check_model("fgsm", "target", target, ("grad",))  # name and grad exist

images = np.random.default_rng(0).integers(0, 256, size=(2, 240, 320, 4), dtype=np.uint8)
check_output("fgsm", target, "grad", images / 255, target.grad(images / 255))  # shape, finite

attacked = Fgsm({"target_model": target, "epsilon": 8 / 255}).apply(images)
print("max change:", np.abs(attacked.astype(int) - images).max(), "grey levels")  # 8
```

Then check that the attack actually affects your model: compare its
predictions on `images` and `attacked`, ideally on real images rather than
random ones. `examples/notebooks/adversarial-frameworks.ipynb` does this for
the three example adapters.

## Choosing the loss

FGSM increases whatever loss `grad` differentiates. The attack only works if that loss
measures what you want to break:

| Task | Loss to return | Effect of the attack |
|---|---|---|
| Classification, no labels | Cross-entropy of the model's own top-1 class (the TensorFlow and JAX examples) | Pushes the prediction away from what the model currently sees |
| Classification, targeted | **Minus** the cross-entropy of a chosen class (a constructor argument, set through `params`) | Pulls the prediction toward that class |
| Detection | **Minus** the total detection confidence (sum of best class scores) | Suppresses detections |
| Segmentation | Pixel-wise cross-entropy of the predicted mask | Corrupts the predicted mask |

Detach anything computed from the model's own prediction (e.g. `top1`), so
the gradient only flows through the loss itself.

`grad` receives only the images, not the dataset's labels, so a loss on each
image's ground-truth label is not possible yet. The model's own prediction is
the usual stand-in, and avoids the "label leaking" effect described in the
FGSM literature.

The notebook `examples/notebooks/adversarial-frameworks.ipynb` shows the
three example adapters at work: the detection loss lowers YOLO's confidence,
and the classifiers' cross-entropy loss flips their top-1 class at 8 grey
levels.

## Pitfalls

- **Put the preprocessing inside the gradient.** Resize and normalize with
  framework ops *after* turning on gradient tracking, so the gradient comes
  back at the full image resolution. Preprocessing done in numpy beforehand
  breaks the gradient chain.
- **Resize with antialiasing.** A plain bilinear downscale (e.g. 1928 → 224
  pixels) only reads a few source pixels, so most pixels get a zero gradient
  and are never attacked. Use `antialias=True` (PyTorch, TensorFlow) or
  `jax.image.resize` (antialiased by default).
- **Evaluation mode.** Call `model.eval()` (PyTorch) or `training=False`
  (Keras), so dropout and batch-norm statistics don't change between calls.
- **Gradient enabled.** Some libraries disable autograd at inference
  (`torch.no_grad`, ultralytics' predictor). Call the underlying network
  yourself under `torch.enable_grad()`. `yolo_target.py` does this for
  ultralytics (`YOLO(weights).model`).
- **Import inside the adapter.** Import your framework inside `__init__` or
  `grad`, not at the top of the file. The framework belongs to your
  environment; kcai itself never depends on it.
- **TensorFlow and PyTorch in one process.** Import `torch`/`torchvision`
  before `tensorflow`: the reverse order crashes Python with tensorflow-cpu
  2.21 and torchvision 0.29.
- **Memory.** `grad` receives the whole transform batch at once. Lower
  `operations.transform_batch_size` if your model runs out of memory.

## Other frameworks

The structure stays the same: inputs to a tensor, preprocess, model, loss,
framework gradient, back to numpy. The core of `grad` in each framework:

=== "PyTorch"

    ```python
    rgb = torch.from_numpy(xs[..., :3].astype(np.float32)).permute(0, 3, 1, 2)
    rgb.requires_grad_(True)
    with torch.enable_grad():
        loss = my_loss(self.model(preprocess(rgb)))
        (dx,) = torch.autograd.grad(loss, rgb)
    dx = dx.permute(0, 2, 3, 1).numpy()
    ```

    Full example: `examples/adapters/yolo_target.py`.

=== "TensorFlow / Keras"

    ```python
    rgb = tf.Variable(xs[..., :3].astype(np.float32))
    with tf.GradientTape() as tape:
        loss = my_loss(self.model(preprocess(rgb), training=False))
    dx = tape.gradient(loss, rgb).numpy()
    ```

    Full example: `examples/adapters/keras_target.py`.

=== "JAX / Flax"

    ```python
    def loss_fn(rgb):
        return my_loss(self.model(preprocess(rgb)))

    dx = np.asarray(jax.grad(loss_fn)(xs[..., :3].astype(np.float32)))
    ```

    For speed, compile it once with `jax.jit`. Full example:
    `examples/adapters/flax_target.py`.

In every case, write `dx` into the RGB planes of a zero array shaped like `xs`
and return it.

## Limits

- **No gradient, no gradient attack.** Inference-only formats and runtimes
  (ONNX Runtime, TensorRT, OpenVINO, TFLite, Core ML), non-differentiable
  models (decision trees, gradient boosting) and remote APIs cannot implement
  `grad`. Give them a `predict` and use the black-box `art_square`, which
  attacks the deployed model itself. Alternatively, write `grad` on the
  training checkpoint the deployed model came from, then check the attacked
  images on the deployed model (a *transfer* attack).
- **Black-box attacks cost queries.** `art_square` calls `predict` up to
  `max_iter` times per image (default 100), stopping early once the top class
  changes: budget the time of one model call times the number of queries.
  It is a random search, so success varies with the seed.
- **Deployed models can behave differently.** Quantized or compiled models
  round differently from their float checkpoint, so an attack computed on the
  checkpoint can be weaker on the deployed model. Measure it on the deployed
  model.
- **Classifiers only, for the black-box attack.** `art_square` needs class
  scores; detection and segmentation models need a `predict` that reduces
  their output to scores (e.g. per-class confidence).

## Troubleshooting

| Error | Cause |
|---|---|
| `fgsm needs a target model exposing grad; 'x' lacks grad` | No callable `grad`, or a typo in its name. |
| `art_square needs a target model exposing predict; 'x' lacks predict` | The black-box attack needs `predict`. |
| `x.predict returned …, expected class scores of shape (B, n_classes)` | `predict` returned one score per image, a tensor, or a single class. |
| `art_square: p_init=… gives a first square of … px` | The first square is larger than the image's short side: lower `p_init`. |
| `the target model must have a non-empty name` | `name` is missing or empty. |
| `x.grad returned …, expected an array of shape (B, H, W, 4)` | `grad` returned a tensor instead of a numpy array, dropped the alpha plane, or kept the channels-first layout. |
| `x.grad returned non-finite values` | NaN/inf in the loss: check the preprocessing (division by zero) and the loss (log of 0). |
| `model 'mine': type: python file not found` | `path` is relative to the directory you run the job from. |
| `exposes several model candidates … set 'export:'` | The file defines more than one adapter-like class: name the one to use with `export`. |
| Images barely change | The loss ignores what you measure, or most pixels get a zero gradient (resize without antialiasing). See [Pitfalls](#pitfalls). |
