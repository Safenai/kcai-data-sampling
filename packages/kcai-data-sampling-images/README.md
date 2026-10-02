# kcai-data-sampling-images

The **procedural** slot of kcai data-sampling: transformations whose output is
computed from the input sample alone. No model enters the computation, so
nothing here needs a framework and nothing here needs a GPU.

These methods work in memory on a decoded `(B, H, W, C)` `uint8` stack, where
`B` is the batch and `(H, W, C)` is one image. They return a batch of the same
shape — the sample space never changes, so the difference between a source
sample and its output stays defined and can be computed downstream.

Its only dependencies are numpy and the core contracts. It deliberately ships no
input/output plugins and no image-file handling: reading and writing are a
separate concern, handled by the job package.

## What it ships today

- `horizontal_flip` — mirror the width axis. Deterministic, and its own
  inverse, so the output determines the input.
- `crop_resize` — keep a window of a chosen fraction and resize it back to the
  original shape. Deterministic and lossy, so it is not invertible.
- `ImageBatch` — a `Batch` specialization for images, where the sample space is
  fixed: axes `(height, width, channel)` and values `(0, 255)`. It adds an
  `images` alias for the stack, so the image math reads naturally.

Two methods, as examples of the slot rather than a catalogue. Both are one
numpy expression in a pure function, wrapped into the transformation contract.

## Install

```bash
pip install "kcai-data-sampling[images]"
```

## Example

```yaml
dataloaders:
  loaders:
    - name: images
      type: parquet
      path: /path/to/samples.parquet
      id_column: id
      decode: img_bytes
      sample_path:
        - column: path
          prefix: /path/to/images

operations:
  outputs:
    path: /path/to/outputs/{selection}/ledger.parquet
    write_samples: true
    samples_dir: /path/to/outputs/{selection}/payloads
  transformations:
    - name: horizontal_flip
      type: horizontal_flip
    - name: crop_resize
      type: crop_resize
      fraction:
        range: [0.3, 0.5]
        step: 0.1
```

Listed transformations are applied **independently** to the same source sample,
not chained: the flip and the crop each get their own row, and neither sees the
other's output. `fraction` is given as a `range` plus a `step`, so the job
expands it into one run per value in the sweep.

Because `crop_resize` resizes the kept window back to the original shape,
every output shares the source's sample space and can be compared against it
pixel by pixel. A method that changed the shape would be refused, not resized.

## Notes

- **The regime is not declared.** Whether an output is a perturbation, an
  augmentation or a corruption is read off the input/output pair afterwards, not
  asserted by the algorithm.
- **`crop_resize` is top-left anchored.** The window is
  `top`/`left` offset by the fraction of `H`/`W`, then resized back by
  nearest-neighbour index replication. Offsets that would leave the image are
  refused rather than silently clamped.
- **Scaling back is lossy by construction.** The crop is kept and the rest
  discarded, so `crop_resize` records itself as not reversible.