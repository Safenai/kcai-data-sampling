# Quick start

End-to-end in a few commands: install the package, fetch the sample data, run
your first job, and read what it produced. The vocabulary used below —
transformation, families, label effects — is defined in
[docs/terminology.md](terminology.md).

## Prerequisites

- Python >= 3.11.

## Install

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Then install the package:

```bash
pip install "kcai-data-sampling[job,images]"
```

The umbrella's base install is the core library alone, and the surfaces you
actually use arrive as extras. `[job]` brings the `process` command and the
parquet dataloader, `[images]` brings the image transformations — between them,
everything the first job below needs. Neither pulls a model framework, so the
result is torch-free.

Verify the install:

```bash
kcai-data-sampling version
kcai-data-sampling list
```

`version` and `list` work on any install; `process` exists only once `-job` is
there. Without it, `version` prints `job: None` and `list` shows empty
categories rather than failing.

The extras are independent, so compose what you need: `[job]` alone for the
CLI, `[images]` alone if you are driving the library from Python, or `[all]` for
every surface plus Jupyter.

## Get the examples

The sample data, the job configurations and the walkthrough notebook live in
the repository rather than in the wheel, so clone it once to follow the rest
of this page:

```bash
git clone https://github.com/Safenai/kcai-data-sampling.git
cd kcai-data-sampling
```

The steps below assume you are at the repository root with your virtual
environment still active.

## Fetch the sample data

```bash
python scripts/fetch_comma10k_sample.py
```

This downloads the ten comma10k frames the tests and the notebook use into
`examples/data/comma10k_sample/` (the agent laces, the sidecar
`samples.parquet`, and a matching `masks/` with the annotation images).

## Run your first job

```bash
kcai-data-sampling process -p examples/config/walkthrough-procedural.yaml
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
with your virtual environment's interpreter as the kernel (`.venv/bin/python`)
and run it top to bottom — the setup cell re-chdirs to the repository root, so
it works from any launch directory.

## Optional surfaces (opt-in)

Two transformations need an extra install; the environment above stays free
of their dependencies. Each has its own umbrella extra:

- **Generative inpainting** (`[lama]`, weights fetched into `.cache/` on first
  use). This is the extra that pulls torch. On Linux the default install gets a
  CUDA build, so on a machine without an NVIDIA GPU ask for the CPU wheel:

  ```bash
  # GPU (default)
  pip install "kcai-data-sampling[lama]"

  # CPU only, no CUDA libraries
  pip install "kcai-data-sampling[lama]" --extra-index-url https://download.pytorch.org/whl/cpu
  ```

  The flavour shows in torch's own version string:
  `python -c "import torch; print(torch.__version__)"` prints a `+cpu` suffix.
  Then run the generative job:

  ```bash
  kcai-data-sampling process -p examples/config/walkthrough-generative.yaml
  ```

- **Adversarial FGSM** (`[fgsm]`, numpy-only in the package; the *target model*
  is your own source with its own weights — the job never ships it):

  ```bash
  pip install "kcai-data-sampling[fgsm]"
  kcai-data-sampling process -p examples/config/walkthrough-adversarial.yaml
  ```

  `walkthrough-adversarial.yaml` names the target adapter file
  (`examples/adapters/yolo_target.py`) and expects your `yolov8n.pt` weights,
  resolved via `ultralytics` on the machine running the job.

To get everything the walkthrough notebook needs in one go, install the `all`
extra — it is the union of `notebooks`, `job`, `images`, `fgsm` and `lama`:

```bash
pip install "kcai-data-sampling[all]"
```

`[all]` pulls torch twice over — directly through `[lama]`, and again through
`ultralytics` in `[notebooks]` — so one flag covers both. As above, add the CPU
index on a machine without an NVIDIA GPU:

```bash
pip install "kcai-data-sampling[all]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Want just the notebook tooling, with no model surface? `[notebooks]` installs
pandas, `python-pptx`, ipykernel and ultralytics, and deliberately does *not*
pull `-fgsm` or `-lama`. Note it does still resolve torch, because ultralytics
depends on it itself — the same GPU/CPU choice applies as for `[lama]` above.

## Where to go next

- [docs/terminology.md](terminology.md) — the vocabulary: families, regimes,
  label effects.
- [docs/developer.md](developer.md) — working on kcai itself: repository
  setup and the test and build sessions.
- [README.md](index.md) — the workspace overview and package layout.
- `examples/notebooks/walkthrough.ipynb` — the full annotated walkthrough.