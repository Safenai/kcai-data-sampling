# Packaging Tests

The six kcai distributions are published separately on purpose: a user who only wants
to run procedural transforms should not download a deep-learning framework. This page
documents how that separation is verified — each scenario installs its own subset into
a fresh virtual environment and proves the environment resolves *exactly* that subset.

Three sources are covered, because they fail differently. Locally built wheels prove
your working tree is packaged correctly. test.pypi.org proves the artifact you actually
uploaded, extras metadata and entry points included. pypi.org proves what a user gets
once the release goes out.

## Why Package Isolation Matters

| Package | Purpose | Notable dependencies |
|---------|---------|----------------------|
| `kcai-data-sampling-core` | Batch, transformations, the model protocols | pydantic, numpy |
| `kcai-data-sampling-images` | Image transforms, the dataloader/writer plugins | Pillow |
| `kcai-data-sampling-job` | The YAML pipeline and its CLI | pyarrow, pyyaml |
| `kcai-data-sampling-fgsm` | FGSM adversarial attack | numpy only — **no torch** |
| `kcai-data-sampling-lama` | LaMa inpainting | torch, but only on first checkpoint load |
| `kcai-data-sampling` | Umbrella CLI plus the `notebook` extra | pulls the others by extra |

Two invariants follow, and both are asserted rather than assumed:

- `kcai-data-sampling-fgsm` must never pull in `torch`. It is a closed-form attack
  against a caller-supplied gradient, so it needs nothing beyond numpy.
- `-lama` must not import `torch` at import time. The framework is needed only when a
  checkpoint is actually loaded, so importing the plugin stays cheap.

Every scenario installs a subset and then probes **all six** module roots. A venv that
resolves a package it did not ask for fails the test, which is how a transitive leak
gets caught.

## Sessions

| Session | Installs from | Marker | Use it |
|---------|---------------|--------|---------|
| `nox -s test_packaging` | locally built wheels | `packaging` | Before every PR — tests your changes |
| `nox -s test_packaging_testpypi` | test.pypi.org | `packaging_index` | After publishing an rc — tests the uploaded artifact |
| `nox -s test_packaging_pypi` | pypi.org | `packaging_index` | After a real release — tests what users get |
| `nox -s test_packaging_notebook` | test.pypi.org | `packaging_notebook` | When the extras or the walkthrough change |

None of these run in the default suite (`nox -s lint spell test type_check`), and
`nox -s test` never even collects them.

```bash
# Every published-index scenario
nox -s test_packaging_testpypi

# One scenario
nox -s test_packaging_testpypi -- -k all+lama

# The real walkthrough notebook (slow, see below)
nox -s test_packaging_notebook
```

The index sessions are **manual, not CI**, for two reasons: test.pypi.org propagates a
newly uploaded file over a few minutes, so a CI job would race the upload; and the
model-backed scenarios pull several hundred megabytes of wheels. Run them after a
publish has settled.

### Which version gets tested

Every distribution is pinned to one version, resolved in this order:

1. `KCAI_PACKAGING_VERSION`, if set
2. otherwise the newest numeric tag, via
   `git describe --tags --match "[0-9]*" --abbrev=0`

The tag is normalized the same way the publish workflows normalize it (`sed 's/-//g'`),
so `0.1.0-rc1` becomes `0.1.0rc1` — exactly the wheel name on the index. Pinning means
a re-run tests the artifact you uploaded rather than whatever else the index may serve
later. To test a specific build:

```bash
KCAI_PACKAGING_VERSION=0.1.0rc1 nox -s test_packaging_testpypi
```

## Scenarios

The first six run against every source. The seventh and the notebook are index-only.

| Scenario | Installs | Smoke script | What it proves |
|----------|----------|--------------|----------------|
| `core` | `-core` | `smoke_core.py` | Batch + transformation runner round-trip; no sibling leaks in |
| `core+images` | `-core`, `-images` | `smoke_images.py` | Image transforms and the image dataloader/writer plugins register |
| `core+job` | `-core`, `-job` | `smoke_job.py` | A synthetic parquet through the CLI — in-memory frames, no dataset |
| `all` | `-core`, `-images`, `-job`, umbrella | `smoke_all.py` | Umbrella `version`/`list`/`process` dispatch works |
| `all+lama` | the four above + `-lama` + CPU torch | `smoke_lama.py` | The model plugin loads and inpaints on a stub tool, still without importing torch |
| `all+fgsm` | the four above + `-fgsm` | `smoke_fgsm.py` | The attack is importable and numpy-only — no torch, no ultralytics |
| `all+notebook` | umbrella`[notebook]` | `smoke_notebook.py` | The extra's third-party deps installed and every plugin is discoverable from wheel metadata |
| `notebook-walkthrough` | umbrella`[notebook]` + nbclient | `execute_walkthrough.py` | The real walkthrough notebook runs end to end |

`all+lama` and `all+notebook` additionally pull `torch` from the PyTorch CPU wheel
index, pinned to `torch==2.14.0+cpu`. Plain `torch` on Linux would otherwise drag in the
multi-gigabyte CUDA build.

`all+notebook` installs **only** the umbrella with the extra, named rather than pointed
at wheels. That is deliberate: it is the published `[notebook]` extra's own dependency
metadata being resolved, not a locally assembled set. Its smoke script probes the extra's
distributions with `importlib.util.find_spec` (top-level names only, so `ultralytics` is
never actually imported and never pulls in `cv2`) and resolves `lama_inpaint`, `inpaint`,
`fgsm`, `horizontal_flip`, `crop_resize`, `parquet`, `images` out of the *installed*
wheel entry points — the same lookup a YAML `type: lama_inpaint` performs.

## The walkthrough notebook

`nox -s test_packaging_notebook` executes `examples/notebooks/walkthrough.ipynb` itself,
in a venv built from the index. It covers all three families: the procedural pipeline
through the CLI, LaMa inpainting behind the `lama_inpaint` entry point, and FGSM against
`examples/adapters/yolo_target.py` loaded through the `python` model route.

It needs the comma10k sample, which the script fetches on first use:

```bash
python scripts/fetch_comma10k_sample.py
```

Two things worth knowing before you run it:

- **It is slow, and it downloads.** Expect minutes and several hundred megabytes on a
  cold cache: CPU torch, the ~200 MB `big-lama.pt` checkpoint, and YOLO weights fetched
  by `ultralytics`. With the checkpoint already in `.cache/` it runs in under two
  minutes.
- **It writes inside the repository**, under `examples/outputs/`, `examples/data/` and
  `.cache/`. Every one of those paths is git-ignored, so `git status` stays clean — but
  `examples/outputs/` grows by ~300 MB per run and nothing prunes it. Delete it
  yourself if you care about disk.

The notebook executor is installed into the test venv only, not added to the published
`[notebook]` extra: the extra targets notebook authors who already have a working Jupyter
install, whereas the executor is the runner's own tool. The practical consequence is that
this test proves the extra is sufficient *except* for an executor.

## Manual testing

For a quick look at what a user would install, without running the suite:

```bash
uv venv --python 3.12 /tmp/kcai
VIRTUAL_ENV=/tmp/kcai uv pip install --no-config --prerelease allow \
    --index-strategy unsafe-best-match \
    --index https://test.pypi.org/simple/ --index https://pypi.org/simple/ \
    kcai-data-sampling==0.1.0rc1
```

Both flags are load-bearing, and dropping either one fails the install:

- `--no-config` — the workspace maps every kcai package to a local path, so without it
  `uv` resolves those instead of the index.
- `--index-strategy unsafe-best-match` — `uv`'s default `first-index` strategy will not
  fall through to the second index. See the gotcha below; it is not optional.

Add `[notebook]` to the spec for the extras (`'kcai-data-sampling[notebook]==0.1.0rc1'`), or
swap the Test PyPI index for `https://pypi.org/simple/` to reproduce a release install.

Two more things about this recipe. `0.1.0rc1` is simply the version published at the
time of writing — substitute whatever is current, remembering that a git tag like
`0.1.0-rc1` normalizes to the PyPI form `0.1.0rc1`. And the recipe assumes no
`UV_INDEX` or `UV_INDEX_URL` in your environment; if you have one set (some setups point
it at an internal registry) you are resolving against a third index you did not intend.
The test harness strips those variables in `_subprocess_env()`; do the same by hand with
`env -u UV_INDEX` if you need to reproduce it exactly.

## Gotchas

- **Just published?** Wait a few minutes. Test PyPI serves the file once it has
  propagated, and a 404 in that window is not a packaging bug.
- **Re-running the same version proves nothing new.** `publish-tolerantly.sh` skips a
  filename that already exists, so a second run does not re-upload. Bump the version (or
  the tag) before expecting the index tests to see new code.
- **`--index-strategy unsafe-best-match` is set on purpose.** `uv`'s default refuses to
  fall through to a second index, and it bites in two separate places. The obvious one is
  direct dependency resolution: seeing any `pydantic` on Test PyPI (1.5a1) hides PyPI's
  2.13 from the resolver. The less obvious one is building source-only distributions —
  `pyyaml` 3.11 has no wheel, so `uv` must build it, and its `setup.py` requires
  `setuptools>=40.8.0`; Test PyPI has a `setuptools` but not a satisfying version, so the
  build fails with *"there are no versions of setuptools and you require
  setuptools>=40.8.0"*. Both indexes here are public and trusted, and every kcai spec is
  pinned to an exact version, so there is no dependency-confusion surface to protect.
- **The scenarios need no dataset.** Frames are synthesized in memory, and the index
  sessions never download data. Only the notebook walkthrough fetches comma10k.
- **The campaign ledger holds one campaign.** Every walkthrough config reuses the
  `comma10k` selection, so each run overwrites `examples/outputs/comma10k/ledger.parquet`
  and only the last campaign's rows survive. That is why the driver asserts on the
  adversarial rows and leaves the rest to the notebook's own cells.