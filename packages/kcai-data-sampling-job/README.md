# kcai-data-sampling-job

Orchestration for kcai data-sampling: everything that turns a *selection* into
files on disk. This package holds no algorithms — it is the layer that runs the
methods other packages provide.

A campaign reads a source table, applies each configured transformation
**independently** to the same source sample, and writes two things per output: a
**ledger** row describing what was produced, and the **payload** file holding the
sample itself. The ledger is metadata only; the pixels go to their own
content-addressed file.

## What it ships today

- `SamplingJob` — the run loop.
- The IO plugins: a `parquet` dataloader, and `images` and `parquet` output
  writers.
- The ledger writer.
- The `process` command, driven by YAML.

**This is where the IO plugins live, and where image files are handled.** It is
also the only part of the workspace that depends on Pillow, and it needs it
because it is the only part that encodes a decoded array to PNG. The
transformation packages themselves stay pure numpy.

## Install

```bash
pip install "kcai-data-sampling[job]"
```

This extra is also what turns on the `process` command:

```bash
kcai-data-sampling process -p config.yaml
```

Without it the command is not there — the umbrella CLI builds its command list
from what is importable, so a base install offers `version` and `list` only.

## How a run works

The job is a streaming loop. For each chunk of `load_batch_size` samples it
decodes a selection into memory, runs every configured transformation over that
chunk through the runner, encodes each output as a payload, buffers the metadata
rows, and drops the chunk.

**Batching changes nothing in the output.** Transformations are unary and the
pairing is chunk-local, so the same selection yields byte-identical results at
any `load_batch_size`. That is why randomness is seeded per parent sample rather
than per batch.

**There is no chain.** Every transformation is applied to the *same* source
sample and each output is its own row — no intermediate stage, no
transformation-of-a-transformation. Two transformations listed together produce
two rows, not one row reached in two steps.

## Output

**The ledger** is one row per generated sample, recording the lineage and the
declaration: which selection and loader it came from, the source `parent_id`, the
output `id`, the algorithm, its `family`, `arity`, `reversible`, the resolved
`params`, the `seed`, and the `tool_model` / `target_model` names. Enough to
reproduce or audit a row without opening the payload.

**Source columns are opt-in and never overwrite the generated ones.** `include`
and `exclude` are pattern lists selecting source columns to carry through;
`exclude` wins over `include`, and any source column sharing a name with a
generated column is dropped from the pass-through rather than allowed to
overwrite it. With neither set, nothing is passed through — the ledger holds
only the generated columns.

**Payload files are content-addressed.** The file name is
`{selection}__{id}__{c6}.png`, where `c6` is a short hash of the decoded payload
bytes. Identical pixels plus the same output id therefore name the same file, so
artifact paths are invariant to buffering and flush order. The output id already
covers the recipe — parent, algorithm, settings, seed — and the hash only guards
content collisions.

## Example

```yaml
dataloaders:
  loaders:
    - name: images
      type: parquet
      path: /path/to/samples.parquet
      id_column: id
      load_batch_size: 10
      decode: img_bytes
      sample_path:
        - column: path
          prefix: /path/to/images

operations:
  outputs:
    path: /path/to/outputs/{selection}/ledger.parquet
    write_samples: true
    samples_dir: /path/to/outputs/{selection}/payloads
    flush_batch_size: 128
    include: [height, width]
  transformations:
    - name: horizontal_flip
      type: horizontal_flip
    - name: crop_resize
      type: crop_resize
      fraction:
        range: [0.3, 0.5]
        step: 0.1
```

`{selection}` in an output path is substituted per selection, so one config can
write several campaigns into separate directories. The `include` list names
source columns to carry into the ledger beyond the generated ones.

## Extending it

The job discovers its parts through entry points, so a package can add a
dataloader or an output writer without changing anything here:

- `kcai_data_sampling.dataloaders`
- `kcai_data_sampling.outputwriter`

Add your own dependencies to *your* package. This one is where IO lives, and
keeping it free of algorithm-specific requirements is what lets a project compose
a run out of any mix of the packages above.