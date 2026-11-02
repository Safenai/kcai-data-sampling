# Changelog

## 0.8.1 (2026-10-30) — Albumentations integration, docs restructure

### Added
- **Albumentations package** (`kcai-data-sampling-albumentation`): 76 robustness transforms across eight families (color/tone, blur/sharpen, noise, geometric, dihedral, occlusion/dropout, weather, compression), configurable from YAML with seeded, replayable application and parameter sweeps.
- **Albumentations test coverage**: unit tests on 6 selected transformations, shared `p` probability tests, and an e2e smoke test covering several transformations.
- **Documentation**: added `docs/usages.md`, added a benchmark comparing albumentations usage through the package vs direct usage, and restructured `mkdocs.yml` navigation.
- **Albumentations README**: added `packages/kcai-data-sampling-albumentation/README.md`.

### Changed
- **Documentation updates**: rewrote the root `README.md` and updated walkthrough configs.
- **Terminology update**: updated `docs/terminology.md` and made the repository coherent with the updated terminology.

## 0.8.0 (2026-10-05) — First release

- **Simple CLI**: Describe a job in a YAML file, run one command. A working example ships with the repository, with a guided walkthrough notebook available.
- **Six packages, one repository**: `kcai-data-sampling-core`, image-handling, job runner, umbrella CLI, and two optional model-backed packages (generative inpainting and adversarial perturbation) — each installable on its own or together.
- **Provenance ledger**: One row per generated sample; generated images are stored separately and addressed by content. Inspect results with any Parquet reader.
- **Image input/output**: Tabular sample descriptions drive image input/output, with mask support across the dataloader, ledger writer, and payload writer.
- **Parameter sweeps**: Request multiple values (e.g. "crop to 10%, 20%, 30%") in a single job.
- **Procedural transformations**: Horizontal flip and crop-and-resize included out of the box.
- **Opt-in model capabilities**: Generative inpainting (LaMa) and adversarial example (FGSM) available as extras.
- **Docs & quality gates**: Documentation site with API reference, quickstart, and glossary; automated tests, formatting, linting, and type checking across Python 3.11–3.14.
