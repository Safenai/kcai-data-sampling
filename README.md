# kcai-data-sampling

Data sampling as a reproducibility pipeline: transformations expand a real
dataset into a sampled one, recording a metadata-only ledger and
content-addressed payload files.

The workspace is a uv monorepo of six member packages under `packages/`:

- `kcai-data-sampling-core` — transformation interface, configuration models, registries and runner;
- `kcai-data-sampling-images` — the procedural family: model-free image transformations (`horizontal_flip`, `crop_resize`) on a decoded image batch;
- `kcai-data-sampling-lama` — generative inpainting (LaMa) transformation and tool-model plugin;
- `kcai-data-sampling-fgsm` — adversarial perturbation (FGSM) transformation and target-model support;
- `kcai-data-sampling-job` — pipeline orchestration and all the IO plugins: the `parquet` dataloader, the `parquet` ledger writer and the `images` payload writer;
- `kcai-data-sampling` — umbrella CLI (`version`, `list`, and `process` once the `[job]` extra is installed).

The vocabulary used across the code and the configs — transformation, families,
regimes, label effects — is defined in [docs/terminology.md](docs/terminology.md).

To get a job running in a few minutes — install, fetch the sample data, run the
procedural example and read its output — see
[docs/quickstart.md](docs/quickstart.md).

## Running the walkthrough notebook

The walkthrough on real comma10k frames lives in
`examples/notebooks/walkthrough.ipynb`; its presentation helpers are a
hand-maintained import (no matplotlib), next to it.

    uv sync --package kcai-data-sampling --extra all
    uv run scripts/fetch_comma10k_sample.py   # 10 frames (+ masks) into examples/data/comma10k_sample/

Then open the notebook in your Jupyter environment (JupyterLab or VS Code) and
pick the workspace interpreter `.venv/bin/python` as the kernel. Run it top to
bottom — the setup cell re-chdirs to the repository root and reads
`examples/config/walkthrough-procedural.yaml` so it works
regardless of the launch directory. Generated files are left under
`examples/outputs/`.