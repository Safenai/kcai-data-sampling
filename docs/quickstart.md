# Quick start

End-to-end in a few commands: install the workspace, fetch the sample data, run
your first job, and read what it produced. The vocabulary used below —
transformation, families, label effects — is defined in
[docs/terminology.md](terminology.md).

## Prerequisites

- Python 3.12 (the workspace pins it in `.python-version`).
- [`uv`](https://docs.astral.sh/uv/).

## Install

```sh
git clone git@github.com:Safenai/kcai-workspace.git
cd kcai-workspace
uv sync
```

The default environment is torch-free: it installs the workspace packages and
the notebook dependencies, but no model heavyweight. Verify the install:

```sh
uv run kcai-data-sampling version
```

## Fetch the sample data

```sh
uv run scripts/fetch_comma10k_sample.py
```

This downloads the ten comma10k frames the tests and the notebook use into
`examples/data/comma10k_sample/` (the agent laces, the sidecar
`samples.parquet`, and a matching `masks/` with the annotation images).

## Run your first job

```sh
uv run kcai-data-sampling process -p examples/config/walkthrough-procedural.yaml
```

A procedural job needs no model: `samples.parquet` is loaded, each frame is
run through the two transformations in the config
(`horizontal_flip`, `crop_resize` at three even fractions), and every output
is written to disk.

## What you get

The job writes a metadata-only ledger and content-addressed payload images:

- `examples/outputs/comma10k/ledger.parquet` — one row per generated sample;
  the generator columns are `selection`, `dataloader`, `parent_id`, `id`,
  `algorithm`, `family`, `arity`, `reversible`, `params`, `seed`,
  `tool_model`, `target_model`, `artifact`.
- `examples/outputs/comma10k/payloads/` — one PNG per `artifact` entry.

Inspect the ledger with pyarrow (no pandas needed):

```python
import pyarrow.parquet as pq

t = pq.read_table("examples/outputs/comma10k/ledger.parquet")
print(t.schema)          # the column layout
print(t.to_pydict())     # the rows, column-arrays
```

Or, instead of the CLI, open `examples/notebooks/walkthrough.ipynb` in Jupyter
with the workspace interpreter (`.venv/bin/python`) as the kernel and run it
top to bottom — the setup cell re-chdirs to the repository root, so it works
from any launch directory.

## Optional surfaces (opt-in)

Two transformations need an extra install; the default environment stays free
of their dependencies:

- **Generative inpainting** (`-lama`, pulls CPU torch, weights fetched into
  `.cache/` on first use):

  ```sh
  uv sync --package kcai-data-sampling-images-lama
  uv run kcai-data-sampling process -p examples/config/walkthrough-generative.yaml
  ```

- **Adversarial FGSM** (`-fgsm`, numpy-only in the package; the *target model*
  is your own source with its own weights — the job never ships it):

  ```sh
  uv sync --group fgsm
  uv run kcai-data-sampling process -p examples/config/walkthrough-adversarial.yaml
  ```

  `walkthrough-adversarial.yaml` names the target adapter file
  (`examples/adapters/yolo_target.py`) and expects your `yolov8n.pt` weights,
  resolved via `ultralytics` on the machine running the job.

## Where to go next

- [docs/terminology.md](terminology.md) — the vocabulary: families, regimes,
  label effects.
- [README.md](../README.md) — the workspace overview and package layout.
- `examples/notebooks/walkthrough.ipynb` — the full annotated walkthrough.