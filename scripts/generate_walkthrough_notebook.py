#!/usr/bin/env python3
"""Generate the phase-1 walkthrough notebook.

Adapts the story of the interface walkthrough (steps 0, 1, 2, 3, 4, 9 and 10)
to the phase-1 interface, keeping the two ways of driving the tool: the
CLI-equivalent ``run(CFG)`` and the core-level API.

``examples/notebooks/display.py`` (imported by the setup cell) is
hand-maintained next to the notebook, not generated.

Usage:
    python scripts/generate_walkthrough_notebook.py
"""

import json
import pathlib

OUT = pathlib.Path("examples/notebooks/walkthrough.ipynb")

INTRO = """# The sample-generation interface, on real driving frames

`T : x ↦ x′`: one fully specified operation: algorithm + resolved parameters + seed. One row per output; no judgement on it.

Runs on ten **comma10k** frames (MIT), fetched by `scripts/fetch_comma10k_sample.py` into `examples/data/comma10k_sample/`. The interface reads the frames; the masks are fetched too and never read — section 1 draws them, the package does not.

**Phase 1, unary procedural:**

| | |
| :--- | :--- |
| unary procedural | `HorizontalFlip`, `CropResize` |

Two ways to drive the same tool — section 6 runs both: the CLI-equivalent `run(CFG)` from a YAML config, and the core-level API by hand.

*No "teardown" cell is included; generated files under `examples/outputs/` are left for the user to manage.*
"""

MD0 = "## 0 · Setup"

CODE0 = """import json, os
from pathlib import Path

import numpy as np
import yaml
from IPython.display import display
from PIL import Image
%config InlineBackend.figure_format = 'jpeg'   # photographs: keeps the committed notebook small

from display import show, rows_table   # presentation only, next to this notebook

# Every path (in the YAML, in the outputs) is relative to the repository root.
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())
os.chdir(REPO)
print("working from the repository root:", REPO)

# --- The phase-1 process config: the one input the CLI way takes ---
with open("examples/config/walkthrough.yaml") as f:
    CFG = yaml.safe_load(f)
print(json.dumps(CFG, indent=2))
"""

MD1 = """## 1 · The samples

A sample is data **and** its annotation. `Sample.y` carries the annotation as the dataset gives it; the interface never reads it. This phase the `image_dir` loader reads the frames only — `y` stays empty, and the masks below are drawn by the notebook, not the package.

Each sample is `(H, W, 4)` uint8 RGBA, decoded once from the folder, in the `0–255` space the selection will declare.
"""

CODE1 = """from kcai_data_sampling_images.input_io import ImageDirDataLoader
from kcai_data_sampling_core.models.dataloaders import DataLoaderConfig

# Built from the YAML's own loader entry: the same config the CLI way will use.
loader_cfg = DataLoaderConfig(**CFG["dataloaders"]["loaders"][0])
loader = ImageDirDataLoader(name=loader_cfg.name, config=loader_cfg)
source = loader.get_selections()[0]                 # streaming: one chunk of load_batch_size at a time

samples = [s for chunk in source for s in chunk]    # materialized: in memory from here on

print(f"{len(samples)} samples, x = {samples[0].x.shape} {samples[0].x.dtype}")
print(f"first: id {samples[0].id!r}, source {samples[0].source}")

# The annotation (comma10k masks): presentation only, here in the notebook
MASKS = Path("examples/data/comma10k_sample/masks")
mask_of = lambda s: np.asarray(Image.open(MASKS / f"{s.id}.png").convert("RGB"))
overlay = lambda x, m, a=0.4: (x[..., :3].astype(np.uint16) * (1 - a) + m.astype(np.uint16) * a).astype(np.uint8)

show([(s.id, s.x) for s in samples[:3]])
show([(s.id, mask_of(s)) for s in samples[:3]])
show([(s.id, overlay(s.x, mask_of(s))) for s in samples[:3]])
"""

MD2 = """## 2 · The data selection

The **dataset** is comma10k; the **data selection** is the ten frames this campaign covers; a **batch** is an execution detail chosen at run time. The selection is the authority on what a sample is: an `id` and a `source` per sample, `sample_axes` for its shape, `value_range` for its domain of values. It lives in memory; writing it is the writers' job (section 6).
"""

CODE2 = """from kcai_data_sampling_core.api.selection import DataSelection
from kcai_data_sampling_core.utils.runner import TransformationRunner

selection = DataSelection(
    name=f"comma10k-{len(samples)}",
    dataset=source.dataset,        # the reader that assembles a sample from its source
    samples=samples,
    sample_axes=source.sample_axes,
    value_range=source.value_range,
)
runner = TransformationRunner(selection)

print(f"{len(selection)} samples; axes {selection.sample_axes}, value_range {selection.value_range}")
"""

MD2B = """Every algorithm works on a batch — `apply` takes `(B, *sample)` and returns `(B, *sample)` as one array operation — and the batch never reaches the row: one batch of ten, two of five, five of two or ten of one give the same rows in the same order. The batch size is a memory choice.
"""

CODE2B = """from kcai_data_sampling_images.transformations.horizontal_flip import HorizontalFlip

flip = HorizontalFlip()
runs = {size: runner.run(flip, batch_size=size) for size in (10, 5, 2, 1)}

reference = runs[10]
for size, results in runs.items():
    same = all(np.array_equal(a.x, b.x) and a.parent_id == b.parent_id and a.params == b.params
               for a, b in zip(results, reference))
    print(f"  batch_size={size:2} → {len(results)} rows, identical to one batch of 10: {same}")
"""

MD3 = """## 3 · What a transformation is

Algorithm, resolved parameters, seed, nothing left open, and that *is* the identity: no separate id on the row. Each algorithm declares its parameters (`CropResize.parameters`); an unknown or a missing one is refused at construction, and the row carries them all, defaults included. The seed rule is relaxed: a seed is always accepted, but a deterministic algorithm draws none — an explicit seed is stored as `None`.
"""

CODE3 = """from kcai_data_sampling_images.transformations.crop_resize import CropResize

a = CropResize({"fraction": 0.4})
b = CropResize({"fraction": 0.4})
c_ = CropResize({"fraction": 0.5})

identity = lambda t: (t.algorithm, json.dumps(t.params, sort_keys=True), t.seed)
same = lambda u, v: np.array_equal(u.transform([samples[0]])[0].x, v.transform([samples[0]])[0].x)
print(f"a vs b  same output {same(a, b)}   same identity {identity(a) == identity(b)}")
print(f"a vs c  same output {same(a, c_)}   same identity {identity(a) == identity(c_)}")
print("declared:", CropResize.parameters, "  resolved on the row:", a.params)

print("no seed given → seed on the row:", flip.transform([samples[0]])[0].seed)
print("seed 7 given  → seed on the row:", HorizontalFlip({"seed": 7}).transform([samples[0]])[0].seed)

for bad in ({}, {"fracton": 0.4}):
    try:
        CropResize(bad)
    except ValueError as e:
        print("refused:", e)
"""

MD4 = """## 4 · Procedural

`HorizontalFlip` mirrors the width axis and is its own inverse. `CropResize` keeps a window and resamples it back to the original size, which keeps `x′` in the same space as `x`: same shape, same `0–255` range.
"""

CODE4 = """zoom = CropResize({"fraction": 0.4, "top": 300, "left": 600})

sample = samples[0]
flipped = runner.run(flip, [sample])[0]     # an Output: the bitmap `x`, the parent's id, and what produced it
zoomed = runner.run(zoom, [sample])[0]

show([("original", sample.x), ("horizontal_flip", flipped.x), ("crop_resize 40 %", zoomed.x)])
rows_table([flipped, zoomed], "one output per sample, every field but the bitmap")
"""

MD5 = """## 5 · What one row carries, and what it does not

Subtraction, not accumulation: a field is on the row only if it can be neither retrieved through a reference nor recomputed afterwards. Section 4's table shows every column of both rows: no measurement, no path — the path arrives with writing (`artifact`), the measurement never does.
"""

MD5B = """No `δ`, no PSNR: a **downstream module** computes them from the row, since `params` and `seed` replay the run. The magnitude says nothing about what was lost: the flip scores as destroyed and is reversible, the crop scores the same and is not.
"""

CODE5B = """def measure(parent, x_prime):
    d = x_prime.astype(np.float64) - parent.x.astype(np.float64)
    mse = float(np.mean(d ** 2))
    return {"delta_linf": float(np.abs(d).max()), "delta_l2": float(np.sqrt((d ** 2).sum())),
            "psnr_db": float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))}

print(f"{'':16} {'δ∞':>8} {'δ2':>9} {'PSNR':>9}   reversible")
for out in (flipped, zoomed):
    m = measure(sample, out.x)
    print(f"{out.algorithm:16} {m['delta_linf']:8.1f} {m['delta_l2']:9.1f} {m['psnr_db']:8.1f}   {out.reversible}")
"""

MD6 = """## 6 · Writing, and how a row leads back

Nothing so far touched the disk: outputs are bitmaps in memory with a reference to their parent. Writing is the job's — two writers, one contract: the **payload writer** encodes each output as a PNG named after the selection, the parent, the algorithm, a hash of `params` and `seed` and a hash of the content; the **ledger writer** merges the rows into one parquet — metadata only, the pixels never live in it. The row on disk is the output minus its bitmap **plus `artifact`**, the file name only storage can fill.

Two ways to drive the same thing:
"""

MD6A = """**The CLI-equivalent** `run(CFG)`, straight from the YAML — the job streams the folder chunk by chunk and chains the transformations (the crop runs on the flipped pixels)"""

CODE6A = """from kcai_data_sampling_job.cli import run

summary = run(CFG)
print("job summary:", summary)

import pandas as pd

out_root = Path("examples/outputs") / CFG["dataloaders"]["loaders"][0]["name"]
print(out_root, ":")
for p in sorted(out_root.iterdir()):
    print("   ", p.name, f"({len(list(p.iterdir()))} files)" if p.is_dir() else "")

ledger = pd.read_parquet(out_root / "ledger.parquet")
print(f"\\n{len(ledger)} rows in the ledger; the chain, on the first frame:")
display(ledger[ledger.parent_id == samples[0].id])
"""

MD6B = """The writers by hand — the **core-level API**, no chain, the crop runs on the original pixels."""

CODE6B = """from kcai_data_sampling_images.output_io import ImagesOutputWriter
from kcai_data_sampling_job.outputwriter.parquet import ParquetOutputWriter

crop = CropResize({"fraction": 0.4})
flip_outputs = runner.run(flip, batch_size=len(selection))
crop_outputs = runner.run(crop, batch_size=len(selection))

payloads = ImagesOutputWriter(
    name="images",
    config={"images_dir": "examples/outputs/{selection}/payloads", "write_images": True},
)
ledger_w = ParquetOutputWriter(
    name="ledger",
    config={"path_pattern": "examples/outputs/{selection}/ledger.parquet", "flush_batch_size": 128},
)

rows = []
for out in (*flip_outputs, *crop_outputs):
    artifact = payloads.write_payload(selection.name, out)
    rows.append({"selection": selection.name, "dataloader": selection.dataset, **out.row(),
                 "params": json.dumps(out.params, sort_keys=True, default=str), "artifact": artifact})
ledger_w.add_rows(selection.name, rows)
ledger_w.flush()

print(f"{len(rows)} rows; payloads and ledger under examples/outputs/{selection.name}/")
"""

MD6C = """From one row and nothing else in memory: `parent_id` → the selection's entry, `artifact` → the PNG. The ledger holds every campaign that shares this root, so the row is picked by its selection. And the flip being a bijection, the row leads back to the parent: undo it and compare.
"""

CODE6C = """row = ledger[(ledger.algorithm == "horizontal_flip") & (ledger.parent_id == sample.id)].iloc[0]

x_prime = np.asarray(Image.open(out_root / "payloads" / row.artifact).convert("RGBA"))
undone = x_prime[:, ::-1]        # HWC: the width axis, flipped back

print(f"row: algorithm {row.algorithm!r}, parent {row.parent_id!r}")
print(f"artifact: {row.artifact}")
print(f"undo the flip, compare to the parent: max |Δ| = {np.abs(undone.astype(int) - sample.x.astype(int)).max()}")
"""

def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "outputs": [],
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    nb = {
        "cells": [
            md_cell(INTRO),
            md_cell(MD0),
            code_cell(CODE0),
            md_cell(MD1),
            code_cell(CODE1),
            md_cell(MD2),
            code_cell(CODE2),
            md_cell(MD2B),
            code_cell(CODE2B),
            md_cell(MD3),
            code_cell(CODE3),
            md_cell(MD4),
            code_cell(CODE4),
            md_cell(MD5),
            md_cell(MD5B),
            code_cell(CODE5B),
            md_cell(MD6),
            md_cell(MD6A),
            code_cell(CODE6A),
            md_cell(MD6B),
            code_cell(CODE6B),
            md_cell(MD6C),
            code_cell(CODE6C),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "python3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build_notebook(), indent=1))
    print(f"Generated {OUT}")


if __name__ == "__main__":
    main()
