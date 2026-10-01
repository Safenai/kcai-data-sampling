# Developer guide

How to set up the repository, run the tests, and build the documentation. If
you only want to *use* kcai, see the [quick start](quickstart.md) instead.

## Prerequisites

- Python >= 3.11 (the package floor; CI also covers 3.12, 3.13 and 3.14).
- [`uv`](https://docs.astral.sh/uv/) — the workspace uses it for every
  environment, including the ones nox creates.
- [`nox`](https://nox.thea.codes/) — installed by `uv sync` (see
  `default-groups` in `pyproject.toml`). Run it as `uv run nox ...`.

## Set up the repository

```bash
git clone git@github.com:Safenai/kcai-data-sampling.git
cd kcai-data-sampling
uv sync
```

`uv sync` creates `.venv/` with all six workspace packages installed in
editable mode. It is deliberately torch-free: neither the CPU torch wheel nor
the ~200 MB LaMa checkpoint is downloaded unless you ask for the opt-in
surfaces below.

Verify:

```bash
uv run kcai-data-sampling version
```

### Opt-in surfaces

The two model-backed surfaces are separate installs so the default environment
stays small:

```bash
uv sync --package kcai-data-sampling-lama   # CPU torch + LaMa
uv sync --group fgsm                        # -fgsm, numpy-only, no torch
```

## Running things with uv

Because the workspace is a uv project with six member packages, `uv run` is
the normal way to execute anything in the development environment:

```bash
uv run kcai-data-sampling process -p examples/config/walkthrough-procedural.yaml
uv run python scripts/fetch_comma10k_sample.py
```

Prefer `uv run <command>` over activating `.venv/` by hand: it resolves the
right interpreter without you having to remember which one it is.

Tests are the exception — run them through nox. `pytest` is not part of the
project environment (`uv sync` installs only the `nox`, `lint` and
`type_check` groups), and each nox session builds its own environment from the
group it needs, so `uv run pytest` will not find the test deps or the workspace
packages.

## Nox sessions

Nox runs each task in its own environment, built from the matching
`[dependency-groups]` entry in the root `pyproject.toml`. List them with:

```bash
uv run nox -l
```

The four marked as selected below are the default set — `uv run nox` with no
arguments runs exactly those.

### Everyday checks

| Command | What it does |
|---|---|
| `uv run nox` | The default set: `lint`, `spell`, `test`, `type_check` |
| `uv run nox -s lint` | ruff lint + `ruff format --check` on `packages/` and `tests/`. Read-only; applies no fixes. |
| `uv run nox -s fmt` | The same checks, but rewriting: sorts imports (dropping unused ones) and formats. This is the session that changes files. |
| `uv run nox -s type_check` | mypy on all six packages. |
| `uv run nox -s spell` | cspell over the whole repository, documentation included. |
| `uv run nox -s complexity` | Cyclomatic complexity of `packages/`; fails above 15. |
| `uv run nox -s test_complexity` | The same check for `tests/`. |

### Tests

| Command | What it does |
|---|---|
| `uv run nox -s test` | Unit, e2e and CLI tests in the default (torch-free, fgsm-free) environment. |
| `uv run nox -s test_coverage` | Every test including the opt-in surfaces, with coverage. Fails under 90%. Writes an HTML report to `docs/reports/htmlcov/` and a pytest report to `docs/reports/pytest/`. |
| `uv run nox -s test_lama` | Only the torch-gated llama tests, in the `-lama` environment. |
| `uv run nox -s test_fgsm` | Only the importorskip-gated fgsm tests, in the `-fgsm` environment. |
| `uv run nox -s compatibility` | Unit and CLI tests on each of Python 3.11, 3.12, 3.13 and 3.14 in turn. |

Every test session seeds `KCAI_TEST_SEED=42`, so runs are reproducible. Pass
extra arguments through to pytest after `--`, for example
`uv run nox -s test -- -k exactly_once -x`. A `-k` filter that matches nothing
makes pytest exit 5, which nox reports as a failed session — not a bug in
your setup.

### Packaging tests

These install real distributions and check that what you uploaded actually
works. `test_packaging` builds the wheels locally; the other three install
from a published index and are slow — see [packaging tests](packaging-tests.md)
for the full procedure.

| Command | What it does |
|---|---|
| `uv run nox -s test_packaging` | Package isolation: fresh venvs, four wheel subsets, smoke scripts. Fast, no network. |
| `uv run nox -s test_packaging_testpypi` | Installs each scenario from test.pypi.org and smoke-tests it. |
| `uv run nox -s test_packaging_pypi` | The same against pypi.org — what a user actually gets. |
| `uv run nox -s test_packaging_notebook` | Installs the `notebook` extra from test.pypi.org and executes the real walkthrough notebook in it. Several hundred MB and minutes of CPU. |

### Documentation

| Command | What it does |
|---|---|
| `uv run nox -s docs_serve` | Serves the site locally with live reload. |
| `uv run nox -s docs_build` | Builds with `--strict`: warnings are errors, so broken internal links and missing nav entries fail the build. |
| `uv run nox -s docs_github_pages` | Builds and pushes the site to GitHub Pages with `gh-deploy --force`. |

`docs_serve` and `docs_github_pages` are pinned to Python 3.12; the others use
whatever the workspace interpreter is.

### Licenses

| Command | What it does |
|---|---|
| `uv run nox -s licenses` | Third-party license report via pip-licenses. Installs main dependencies only, so the report reflects what users pull in rather than the dev toolchain. |

## What the docs build does to your tree

`docs_build` and `docs_serve` run the hooks in `docs/hooks.py`, which generate
files inside `docs/` before the build: `index.md` from `README.md`,
`CHANGELOG.md` and `RELEASE.md` from the repository root, the six package
READMEs under `docs/packages/`, and the example notebooks under
`docs/examples/`. All of them are git-ignored, so `git status` stays clean —
but do not edit the generated copies, edit the sources.

The hooks also rewrite relative links for the site. A `docs/quickstart.md`
link is correct in a repository-root file and wrong in the copy under `docs/`,
so `_rebase_docs_links()` strips the prefix on copy. Editing a generated file
means your change disappears on the next build.

## Repository layout

```
packages/          the six distributions, each with its own pyproject.toml
  kcai-data-sampling{,-core,-images,-lama,-fgsm,-job}/
tests/             unit, e2e, cli and packaging suites
examples/          configs, adapters, notebooks and (generated) outputs
docs/              this site
scripts/           sample-data fetcher and maintenance helpers
noxfile.py         every session defined here
```

## Before you open a pull request

```bash
uv run nox            # lint, spell, test, type_check
uv run nox -s docs_build
```

If you touched the walkthrough notebook, the package metadata or the extras,
also run the relevant packaging session against Test PyPI and note the version
you tested — [packaging tests](packaging-tests.md) explains how to pick one.