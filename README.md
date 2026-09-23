# kcai-workspace

Data sampling as a reproducibility pipeline: transformations expand a real
dataset into a sampled one, recording a metadata-only ledger and
content-addressed payload files.

The workspace is a uv monorepo of four member packages under `packages/`:

- `kcai-data-sampling-core` — transformation interface, configuration models, registries and runner;
- `kcai-data-sampling-images` — image input/output plugins and image transformations;
- `kcai-data-sampling-job` — pipeline orchestration, parquet dataloader and ledger writer;
- `kcai-data-sampling` — umbrella CLI (`version`, `list`, `process`).

The vocabulary used across the code and the configs — transformation, families,
regimes, label effects — is defined in [docs/terminology.md](docs/terminology.md).

## Running the walkthrough notebook

The phase-1 story on real comma10k frames lives in
`examples/notebooks/walkthrough.ipynb`; its presentation helpers are a
hand-maintained import (no matplotlib), next to it.

    uv sync                                   # four packages + notebook deps into .venv/
    uv run scripts/fetch_comma10k_sample.py   # 10 frames (+ masks) into examples/data/comma10k_sample/

Then open the notebook in your Jupyter environment (JupyterLab or VS Code) and
pick the workspace interpreter `.venv/bin/python` as the kernel. Run it top to
bottom — the setup cell re-chdirs to the repository root and reads
`examples/config/walkthrough.yaml`, so it works regardless of the launch
directory. Generated files are left under `examples/outputs/`.