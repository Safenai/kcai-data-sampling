# kcai-data-sampling

The command-line front door to kcai data-sampling, and the place its pieces are
composed.

A **transformation** is one fully specified operation,

    T : x ↦ x'

taking one input sample to one output sample in the same space — an algorithm,
its resolved parameters, a seed where it draws randomness, and a model where one
is involved. This package installs the core contracts that define that operation
and the CLI that inspects it. Every method, and every piece of IO, lives in a
separate package you add yourself.

## What it ships today

- `version` — the installed versions of the umbrella, the core, and the
  `job` / `images` members when present.
- `list` — every registered plugin, grouped into transformations, models,
  dataloaders and output writers.
- `process` — run a sampling job from a YAML configuration. **Available only
  when the job package is installed**; see the `[job]` extra below.

## Install

```bash
pip install kcai-data-sampling
```

That is the base install, and it is deliberately small: the core library plus the
CLI. It resolves to two distributions and nothing more.

**A consequence worth knowing:** because the CLI builds its command list from
what is actually importable, a base install gives you a working `kcai-data-sampling`
with `version` and `list` — and no `process`. That is not a broken install; it is
the designed default. `version` reports the absent members rather than hiding
them:

```
kcai-data-sampling: 0.1.0
core: 0.1.0
job: None
images: None
```

Add the extra for the surface you want and the matching command appears.

## Extras

| Extra | Adds | Notes |
| --- | --- | --- |
| `job` | the job package: the `process` command, the `parquet` dataloader, and the `images` and `parquet` output writers | the YAML pipeline; also where image-file handling and Pillow live |
| `images` | `horizontal_flip` and `crop_resize`, plus the image batch type | numpy only |
| `fgsm` | the `fgsm` attack | numpy only, no torch |
| `lama` | `inpaint` and the `lama_inpaint` tool model | pulls torch; the checkpoint is fetched on first use |
| `notebooks` | pandas, `python-pptx`, ipykernel and ultralytics — for reading and plotting results | installs **no** `fgsm` and **no** `lama`; still resolves torch, because ultralytics depends on it |
| `all` | every row above | the recipe to use if you want the whole set |

Install one, or combine them:

```bash
pip install "kcai-data-sampling[job,images]"
pip install "kcai-data-sampling[all]"
```

The extras are independent — asking for `[job]` installs the job package and
leaves the image methods out — so a project can compose exactly the surface it
needs. `kcai-data-sampling version` is the quickest way to see what a given
install actually resolved to.

## The three families

Methods are grouped by **the model's role**, not chosen by hand:

| Model role | Family | Meaning |
| --- | --- | --- |
| no model | procedural | computed from the input sample alone |
| tool | generative | the model produces the output content |
| target | adversarial | the model's response enters the computation |

Two more properties hold across all of them. Transformations listed together are
applied **independently** to the same source sample, not chained. And whether an
output is a perturbation, an augmentation or a corruption is read off the
input/output pair *afterwards* — it is not declared by the algorithm.

## Running a job

With the `job` extra installed:

```bash
kcai-data-sampling process -p config.yaml
```

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
  transformations:
    - name: horizontal_flip
      type: horizontal_flip
```

Each run writes a **ledger** — one metadata-only row per generated sample,
recording its lineage and the algorithm's declaration — and the payloads beside
it as content-addressed files.

## Where to go next

The sibling packages each document one thing: the core contracts and how to add
your own method, the procedural slot, the job's orchestration, and the
generative and adversarial slots. Their names are the distribution names, and
each is installed through the matching extra above.