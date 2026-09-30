# kcai-data-sampling 0.1.0 — first release

> Generate more data from the data you already have — and keep a receipt for
> every sample you make.

## What this is

`kcai-data-sampling` is a toolkit for **data sampling**: taking a real dataset and
expanding it into a larger, sampled one.

The part that matters is the receipt. Every generated sample is written to disk
with a short record of exactly how it was produced — what it came from, which
operation was applied, with which settings, and with which random seed. Six
months later you can still answer "why does this image exist?" without guessing,
without re-running anything, and without keeping a sprawling folder of
sidecar files.

In practice that means you can grow a training or evaluation set, hand it to
someone else, and have full confidence they are looking at the same data you
are.

## Why it is built this way

**Reproducible by construction, not by discipline.** A sample's identity is a
fingerprint of its recipe — its parent, the operation, the resolved settings,
the seed. Two runs of the same job produce byte-identical output. Provenance is
not a step you remember to do; it is the only way the system can work.

**Three kinds of expansion, one interface.** Useful data generation falls into
three families, and they behave very differently:

| Family | What it does | Example |
| --- | --- | --- |
| **Procedural** | Classic edits, no model involved | crop, flip, blur, noise |
| **Generative** | A model invents the content | inpainting a removed object |
| **Adversarial** | Crafted *against* a model to stress it | a perturbation that fools a detector |

The first is cheap and safe to run anywhere. The second needs real compute. The
third needs a model to attack. Rather than hide that behind three different
tools, `kcai-data-sampling` gives them one shared shape, so a job can mix them and
so you always know which kind you got.

**Nothing is hardwired.** Transformations, data loaders, and models are all
plugins. Point the tool at your own code, your own model, your own file format,
and it will record the results the same way.

**The default install stays small.** The out-of-the-box environment pulls no
heavy frameworks. Model-based capabilities are opt-in extras you choose
deliberately.

## What is in this release

- **A simple command line.** Describe a job in a YAML file, run one command.
  A working example ships with the repository, and there is a guided
  walkthrough notebook if you would rather read than script.
- **Six packages, one repository**, each installable on its own or together:
  the shared core, an image-handling package, a job runner, an umbrella CLI, and
  two optional model-backed packages (generative inpainting and adversarial
  perturbation).
- **A provenance ledger.** One row per generated sample; the generated images
  themselves are stored separately, addressed by content. Inspect the result
  with any Parquet reader.
- **Image input and output** from a standard tabular sample description, with
  masks supported.
- **Parameter sweeps** — ask for "crop to 10%, 20%, 30%" in one line rather
  than writing three jobs.
- **Procedural transformations** included: horizontal flip and crop-and-resize.
- **Generative inpainting** (LaMa) and **adversarial perturbation** (FGSM)
  available as opt-in extras.
- **A documentation site** with API reference, a quick start, and a glossary of
  the vocabulary.
- **Quality gates in continuous integration** — automated tests, formatting,
  linting, and type checking run on every change across Python 3.11 to 3.14.

## Try it

Four commands, about five minutes, and a small public sample dataset
downloaded for you:

```sh
git clone git@github.com:Safenai/kcai-data-sampling.git
cd kcai-data-sampling
uv sync
uv run kcai-data-sampling version
uv run scripts/fetch_comma10k_sample.py
uv run kcai-data-sampling process -p examples/config/walkthrough-procedural.yaml
```

You end up with a ledger of what was generated and a folder of images, and you
can look at both.

The full walkthrough is in [docs/quickstart.md](docs/quickstart.md); the
vocabulary — *transformation, family, regime, label effect* — is defined
plainly in [docs/terminology.md](docs/terminology.md).

## Before you upgrade

This is the first release, so there is nothing to upgrade from. Two things worth
knowing:

- **Python 3.11 or newer** is required. Python 3.10 reached end of life in
  October 2026 and is no longer supported.
- **The API is still settling.** Treat configuration files you write now as
  something to expect to adjust.

## What comes next

TODO
