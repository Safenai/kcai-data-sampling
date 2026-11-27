# kcai-data-sampling

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/license/apache-2-0)
![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)
[![Nox](https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg)](https://nox.thea.codes/en/stable/)

## Available on PyPI

Install individual packages based on your needs:

| Package | Description | PyPI |
|---------|-------------|------|
| **kcai-data-sampling-core** | Transformation interface, configuration models, registries and runner | [![][pypi-core-badge]](https://pypi.org/project/kcai-data-sampling-core/) |
| **kcai-data-sampling-images** | Procedural family: model-free image transformations | [![][pypi-images-badge]](https://pypi.org/project/kcai-data-sampling-images/) |
| **kcai-data-sampling-albumentation** | Curated image-robustness catalogue (albumentations) | [![][pypi-albumentation-badge]](https://pypi.org/project/kcai-data-sampling-albumentation/) |
| **kcai-data-sampling-lama** | Generative inpainting (LaMa) transformation and tool-model plugin | [![][pypi-lama-badge]](https://pypi.org/project/kcai-data-sampling-lama/) |
| **kcai-data-sampling-fgsm** | Adversarial (FGSM) transformation and target-model support | [![][pypi-fgsm-badge]](https://pypi.org/project/kcai-data-sampling-fgsm/) |
| **kcai-data-sampling-job** | Pipeline orchestration and IO plugins | [![][pypi-job-badge]](https://pypi.org/project/kcai-data-sampling-job/) |
| **kcai-data-sampling-timeseries** | No-model time-series transformations and window batches | [![][pypi-timeseries-badge]](https://pypi.org/project/kcai-data-sampling-timeseries/) |

> **Note:** The `kcai-data-sampling` package is the CLI wrapper. Install
> `[timeseries]` for the algorithms, and add `[job]` for YAML runs on windows.

[pypi-core-badge]: https://img.shields.io/pypi/v/kcai-data-sampling-core.svg
[pypi-images-badge]: https://img.shields.io/pypi/v/kcai-data-sampling-images.svg
[pypi-albumentation-badge]: https://img.shields.io/pypi/v/kcai-data-sampling-albumentation.svg
[pypi-lama-badge]: https://img.shields.io/pypi/v/kcai-data-sampling-lama.svg
[pypi-fgsm-badge]: https://img.shields.io/pypi/v/kcai-data-sampling-fgsm.svg
[pypi-job-badge]: https://img.shields.io/pypi/v/kcai-data-sampling-job.svg
[pypi-timeseries-badge]: https://img.shields.io/pypi/v/kcai-data-sampling-timeseries.svg
[pypi-cli-badge]: https://img.shields.io/pypi/v/kcai-data-sampling.svg

## Documentation

- [Home](https://safenai.github.io/kcai-data-sampling/) — project overview
- [Quick Start](docs/quickstart.md) — install, fetch sample data, run the procedural example, read its output.
- Examples — two notebooks: the [image walkthrough](examples/notebooks/image-walkthrough.ipynb) (procedural, LaMa, FGSM on comma10k frames) and the [time-series walkthrough](examples/notebooks/timeseries-walkthrough.ipynb) (windows, transformations, forecasting and anomaly detection on ETT).
- [Terminology](docs/terminology.md) — vocabulary used across code and configs (transformation, families, regimes, annotation states).
- [Usages](docs/usages.md) — user scenarios (robustness testing, controlled context, composite) and benefits.
- [Benchmark](docs/benchmark/albumentations.md) — albumentations through our package vs. direct.
- [Release Notes](RELEASE.md)
- [Developer Guide](docs/developer.md) — developer notes; see also [Packaging Tests](docs/packaging-tests.md).
- [Packages](docs/packages/kcai-data-sampling-core.md) — API docs for core, images, lama, fgsm, albumentations, job, time series, and cli.

## What is KCAI Sampling?

**KCAI Data-Sampling** is an open-source Python library that helps you
**create and control the data your models are trained and evaluated on**.
Whether you're testing model robustness, building contextual validation sets,
or stress-testing against adversarial inputs, it lets you generate targeted
variants of a real dataset (procedural transforms, generative completions,
adversarial samples) and keep a reproducible record of how each was
produced.

Think of it as a **controlled data generator** — every sample comes with its
recipe, so experiments can be reproduced, compared, and audited.

## Why use it

- **Configuration-driven data selection**: Run transformations on specific data
  selections via a configuration (no need to write ad-hoc scripts per experiment).
- **Reproducible recipes**: Every run produces a recipe in the output, making
  experiments traceable and reproducible.
- **Three composable families**: Procedural, generative, and adversarial
  transformations are all supported in a single tool, enabling complex data
  science scenarios that combine two or more transformation families.
- **Unified workflow**: Switch between robustness testing and
  controlled-context generation without changing tools.
- **Extensible plugin model**: New transformation families/packages can be
  integrated without redesigning the orchestration/config model. The system
  supports input and output plugins, allowing custom data sources and
  destinations to be added.
- **Contract-driven outputs**: Standardized outputs make it easier to compare
  results across transformation families and runs.

## Installation

```bash
pip install "kcai-data-sampling[all]"
```

Select only the extras you need:

```bash
pip install "kcai-data-sampling[core,images,job]"
```

For time-series windows, install `[timeseries]` alone for the algorithms or add
`[job]` for the YAML pipeline:

```bash
pip install "kcai-data-sampling[job,timeseries]"
```

`[all]` pulls torch, which on Linux defaults to a CUDA build. On a machine
without an NVIDIA GPU, ask for the CPU wheel instead:

```bash
pip install "kcai-data-sampling[all]" --extra-index-url https://download.pytorch.org/whl/cpu
```

For development (editable install with all extras and dev tools):

```bash
git clone https://github.com/Safenai/kcai-data-sampling.git
cd kcai-data-sampling
uv sync --all-extras
```

## Quick start (CLI)

```bash
kcai-data-sampling process -p examples/config/walkthrough-procedural.yaml
```

Available subcommands: `version`, `list`, `process`. `process` requires the
`[job]` extra to be installed. For programmatic use, see the
[quickstart](docs/quickstart.md).

## Workspace structure

```
kcai-data-sampling/
├── packages/          # member packages (core, images, albumentation, lama, fgsm, job, timeseries, cli)
├── examples/          # configs, notebooks, sample data
├── docs/              # documentation (quickstart, terminology, usages, benchmark, ...)
├── benchmark/         # benchmark harness (s1/s2/s3)
├── scripts/           # helper scripts (fetch sample data, code quality, ...)
├── tests/             # unit and e2e tests
└── noxfile.py         # lint, type_check, test, coverage sessions
```

## Running the example notebooks

`examples/notebooks/` holds two notebooks: `image-walkthrough.ipynb` and
`timeseries-walkthrough.ipynb`.

### Image walkthrough

The image walkthrough runs the procedural job, LaMa inpainting and FGSM on real
comma10k frames. Its notebook, configs and data all live in the repository
rather than in the wheel, so if you have not cloned it yet:

```bash
git clone https://github.com/Safenai/kcai-data-sampling.git
cd kcai-data-sampling
```

If you have not been through the [quickstart](docs/quickstart.md), set up an
environment first:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Then install the `all` extra — the union of everything the notebook touches —
and fetch the ten sample frames:

```bash
pip install "kcai-data-sampling[all]"
python scripts/fetch_comma10k_sample.py
```

`[all]` pulls torch, which on Linux defaults to a CUDA build. On a machine
without an NVIDIA GPU, ask for the CPU wheel instead:

```bash
pip install "kcai-data-sampling[all]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Open `examples/notebooks/image-walkthrough.ipynb` in Jupyter (JupyterLab or VS Code)
and pick the interpreter of the environment you activated as the kernel. Run it
top to bottom — the setup cell re-chdirs to the repository root and reads
`examples/config/walkthrough-procedural.yaml`, so it works regardless of the
launch directory. Generated files are left under `examples/outputs/`.

### Time-series walkthrough

The time-series walkthrough runs on ETT, two power transformers logged hourly:
it cuts one-week windows, shows what each transformation does, then measures a
forecaster and an anomaly detector on the generated test sets. From a clone,
with the workspace set up (`uv sync`, see the [developer guide](docs/developer.md)):

```bash
uv run python scripts/fetch_ett_sample.py
uv run --with scikit-learn --with matplotlib --with ipykernel --with jupyterlab \
    jupyter lab examples/notebooks/timeseries-walkthrough.ipynb
```

ETT is licensed CC BY-ND 4.0: the data is downloaded by the script, never
shipped, and no modified data (CSV, parquet, payloads) is committed. The
notebook is committed with its outputs, which are plots and summary tables
only.

## Companion tool

[dqm-ml](https://github.com/Safenai/dqm-ml-workspace) — a "health check for
your data". Use it to measure the **diversity**, **representativeness**, or
**domain gap** of a dataset before deciding what to sample or augment. Identify
which contexts or features are under-covered, then target those gaps with
kcai-sampling. Install with `pip install "dqm-ml[all]"`.