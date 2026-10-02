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

The walkthrough runs the procedural job, LaMa inpainting and FGSM on real
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

Open `examples/notebooks/walkthrough.ipynb` in Jupyter (JupyterLab or VS Code)
and pick the interpreter of the environment you activated as the kernel. Run it
top to bottom — the setup cell re-chdirs to the repository root and reads
`examples/config/walkthrough-procedural.yaml`, so it works regardless of the
launch directory. Generated files are left under `examples/outputs/`.