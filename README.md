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