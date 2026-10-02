#!/usr/bin/env python3
"""Generate the walkthrough notebook.

Adapts the story of the interface walkthrough (steps 0, 1, 2, 3, 4, 9 and 10)
to the current interface, keeping the two ways of driving the tool: the
CLI-equivalent ``run(CFG)`` and the core-level API. The procedural sections
drive a parameter range (sweep) through the CLI on a raw-bytes sidecar table;
the last section runs a generative inpainting job through the CLI on the same
comma10k sample (needs the opt-in ``-lama`` install with torch).

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

Runs on ten **comma10k** frames (MIT), fetched by `scripts/fetch_comma10k_sample.py` into `examples/data/comma10k_sample/` — parquet is the only input, so the frames land as a sidecar table, `samples.parquet`, one row per frame. The interface reads the frames; the masks are fetched too and never read — section 1 draws them, the package does not.

**Unary procedural:**

| | |
| :--- | :--- |
| unary procedural | `HorizontalFlip`, `CropResize` |
| parameter range | `fraction: {range, samples, mode}` → one instance per value |

Two ways to drive the same tool — sections 6 and 7 run both: the CLI-equivalent `run(CFG)` from a YAML config, and the core-level API by hand.

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

# --- The process config: the one input the CLI way takes ---
with open("examples/config/walkthrough-procedural.yaml") as f:
    CFG = yaml.safe_load(f)
print(json.dumps(CFG, indent=2))
"""

MD1 = """## 1 · The samples

A sample is a **row**: `id`, `height`, `width`, `path` — and the image itself, decoded **once** at load. The fetch script writes the sidecar `samples.parquet` beside the frames, and the `path` column (relative, resolved against `sample_path.prefix`) names each image. The loader turns each `load_batch_size` chunk into one in-memory **batch**: the source columns stay a table, the image column becomes one numpy `(B, H, W, 4)` uint8 stack — row `i` on every axis is the same sample.

There is no `y`: the interface never reads an annotation. The comma10k masks below are drawn by the notebook, not the package.
"""

CODE1 = """from kcai_data_sampling_job.dataloaders.api.parquet import (
    ParquetDataLoader,
    ParquetImageLoaderConfig,
)

# Built from the YAML's own loader entry: the same config the CLI way will use.
loader_cfg = ParquetImageLoaderConfig(**CFG["dataloaders"]["loaders"][0])
loader = ParquetDataLoader(name=loader_cfg.name, config=loader_cfg)
selection = loader.get_selections()[0]       # one selection per loader, named <loader>
selection.bootstrap(None)
print(selection)

batch = next(iter(selection))                # load_batch_size=10: the whole table, one batch

print(f"{len(batch)} rows; source columns {batch.columns.column_names}")
print(f"data: {batch.data.shape} {batch.data.dtype}; first ids {batch.ids[:3]}; "
      f"space {batch.sample_axes} in {batch.value_range}")

# The annotation (comma10k masks): presentation only, here in the notebook
MASKS = Path("examples/data/comma10k_sample/masks")
mask_of = lambda rid: np.asarray(Image.open(MASKS / f"{rid}.png").convert("RGB"))
overlay = lambda x, m, a=0.4: (x[..., :3].astype(np.uint16) * (1 - a) + m.astype(np.uint16) * a).astype(np.uint8)

show([(batch.ids[i], batch.data[i]) for i in range(3)], width=480)
show([(batch.ids[i], mask_of(batch.ids[i])) for i in range(3)], width=480)
show([(batch.ids[i], overlay(batch.data[i], mask_of(batch.ids[i]))) for i in range(3)], width=480)
"""

MD2 = """## 2 · The data selection

The **dataset** is comma10k; the **data selection** is the ten rows this campaign covers — a (filtered) view of the table, owned by the loader and named `<loader>`; a **batch** is an execution detail chosen at run time. The batch is the in-memory authority on what a sample is: `ids`, the source `columns`, the decoded `data`, and the space every row lives in — `sample_axes` for its shape, `value_range` for its domain of values.
"""

CODE2 = """from kcai_data_sampling_core.utils.runner import TransformationRunner

runner = TransformationRunner(batch)

print(f"selection {batch.name!r} by loader {batch.dataset!r}: {len(batch)} rows")
print(f"axes {batch.sample_axes}, value_range {batch.value_range}")
"""

MD2B = """Every algorithm works on a batch — `apply` takes `(B, *sample)` and returns `(B, *sample)` as one array operation — and the batch never reaches the row: one batch of ten, two of five, five of two or ten of one give the same rows in the same order. The batch size is a memory choice.
"""

CODE2B = """from kcai_data_sampling_core.api.selection import Batch
from kcai_data_sampling_images.api.transformations.horizontal_flip import HorizontalFlip

def subset(b, rows):                    # the same rows, re-batched: still one batch
    return Batch(name=b.name, dataset=b.dataset, ids=[b.ids[i] for i in rows],
                 columns=b.columns.take(rows), data=b.data[rows],
                 sample_axes=b.sample_axes, value_range=b.value_range)

flip = HorizontalFlip()
partitions = {"one batch of 10": [list(range(10))],
              "two of 5": [list(range(0, 5)), list(range(5, 10))],
              "five of 2": [list(range(i, i + 2)) for i in range(0, 10, 2)],
              "ten of 1": [[i] for i in range(10)]}

reference = [o for rows in partitions["one batch of 10"] for o in runner.run(flip, subset(batch, rows))]
for label, chunks in partitions.items():
    results = [o for rows in chunks for o in runner.run(flip, subset(batch, rows))]
    same = all(np.array_equal(a.x, b.x) and a.parent_id == b.parent_id and a.params == b.params
               for a, b in zip(results, reference))
    print(f"  {label:16} → {len(results)} rows, identical to one batch of 10: {same}")
"""

MD3 = """## 3 · What a transformation is

Algorithm, resolved parameters, seed, nothing left open, and that *is* the identity: no separate id on the row. Each algorithm declares its parameters once, as a pydantic schema on `CropResize.Config`; an unknown or a missing one is refused at construction, and the row carries them all, defaults included. The seed rule is relaxed: a seed is always accepted, but a deterministic algorithm draws none — an explicit seed is stored as `None`.
"""

CODE3 = """from kcai_data_sampling_images.api.transformations.crop_resize import CropResize

one = batch.row(0)                     # a single-row batch: the sample at index 0

a = CropResize({"fraction": 0.4})
b = CropResize({"fraction": 0.4})
c_ = CropResize({"fraction": 0.5})

identity = lambda t: (t.algorithm, json.dumps(t.params, sort_keys=True), t.seed)
same = lambda u, v: np.array_equal(u.transform(one)[0].x, v.transform(one)[0].x)
print(f"a vs b  same output {same(a, b)}   same identity {identity(a) == identity(b)}")
print(f"a vs c  same output {same(a, c_)}   same identity {identity(a) == identity(c_)}")
print("declared:", sorted(CropResize.Config.model_fields), "  resolved on the row:", a.params)

print("no seed given → seed on the row:", flip.transform(one)[0].seed)
print("seed 7 given  → seed on the row:", HorizontalFlip({"seed": 7}).transform(one)[0].seed)

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

parent = batch.data[0]
flipped = runner.run(flip, batch=one)[0]   # an Output: the bitmap x, the parent's id, and what produced it
zoomed = runner.run(zoom, batch=one)[0]

show([("original", parent), ("horizontal_flip", flipped.x), ("crop_resize 40 %", zoomed.x)], width=480)
rows_table([flipped, zoomed], "one output per row, every field but the bitmap")
"""

MD5 = """## 5 · What one row carries, and what it does not

Subtraction, not accumulation: a field is on the row only if it can be neither retrieved through a reference nor recomputed afterwards. Section 4's table shows every column of both rows: no measurement, no path — the path arrives with writing (`artifact`), the measurement never does.
"""

MD5B = """No `δ`, no PSNR: a **downstream module** computes them from the row, since `params` and `seed` replay the run. The magnitude says nothing about what was lost: the flip scores as destroyed and is reversible, the crop scores the same and is not.
"""

CODE5B = """def measure(parent, x_prime):
    d = x_prime.astype(np.float64) - parent.astype(np.float64)
    mse = float(np.mean(d ** 2))
    return {"delta_linf": float(np.abs(d).max()), "delta_l2": float(np.sqrt((d ** 2).sum())),
            "psnr_db": float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))}

print(f"{'':16} {'δ∞':>8} {'δ2':>9} {'PSNR':>9}   reversible")
for out in (flipped, zoomed):
    m = measure(parent, out.x)
    print(f"{out.algorithm:16} {m['delta_linf']:8.1f} {m['delta_l2']:9.1f} {m['psnr_db']:8.1f}   {out.reversible}")
"""

MD6 = """## 6 · Writing, and how a row leads back

Nothing so far touched the disk: outputs are bitmaps in memory with a reference to their parent. Writing is the job's — two writers, one contract: the **payload writer** encodes each output as a PNG named after the selection, the output's id and a hash of its content, buffering the bytes in memory and writing them at `flush`; the **ledger writer** merges the rows into one parquet — metadata only, the pixels never live in it. The row on disk is the output minus its bitmap **plus `artifact`**, the file name only storage can fill.

Two ways to drive the same thing:
"""

MD6A = """**The CLI-equivalent** `run(CFG)`, straight from the YAML — the job streams the table chunk by chunk and applies each transformation independently (the crop and the flip both start from the original pixels)"""

CODE6A = """from kcai_data_sampling_job.cli import run

# Reload the config: this cell must work even if only it was re-run after a YAML edit.
with open("examples/config/walkthrough-procedural.yaml") as f:
    CFG = yaml.safe_load(f)

summary = run(CFG)
print("job summary:", summary)

import pandas as pd

out_root = Path("examples/outputs") / CFG["dataloaders"]["loaders"][0]["name"]
print(out_root, ":")
for p in sorted(out_root.iterdir()):
    print("   ", p.name, f"({len(list(p.iterdir()))} files)" if p.is_dir() else "")

ledger = pd.read_parquet(out_root / "ledger.parquet")
print(f"\\n{len(ledger)} rows in the ledger; the outputs, on the first frame:")
display(ledger[ledger.parent_id == batch.ids[0]])
"""

MD6B = """The writers by hand — the **core-level API**."""

CODE6B = """from kcai_data_sampling_job.outputwriter import ImagesOutputWriter, ParquetOutputWriter

crop = CropResize({"fraction": 0.4})
flip_outputs = runner.run(flip)
crop_outputs = runner.run(crop)

hand = f"{batch.name}-by-hand"         # a second campaign, not the CLI's selection
payloads = ImagesOutputWriter(
    name="images",
    config={"samples_dir": "examples/outputs/{selection}/payloads", "write_samples": True,
            "flush_batch_size": 5},
)
ledger_w = ParquetOutputWriter(
    name="ledger",
    config={"path_pattern": "examples/outputs/{selection}/ledger.parquet"},
)

rows = []
for out in (*flip_outputs, *crop_outputs):
    artifact = payloads.add_payload(hand, out)     # encoded now, buffered — no file yet
    rows.append({"selection": hand, "dataloader": batch.dataset, **out.row(),
                 "params": json.dumps(out.params, sort_keys=True, default=str), "artifact": artifact})
ledger_w.add_rows(hand, rows)
payloads.flush()                                   # the payload files land here, content-addressed
ledger_w.flush()

print(f"{len(rows)} rows; payloads and ledger under examples/outputs/{hand}/")
"""

MD6C = """From one row and nothing else in memory: `parent_id` → the selection's entry, `artifact` → the PNG. The ledger holds every campaign that shares this root, so the row is picked by its selection. And the flip being a bijection, the row leads back to the parent: undo it and compare.
"""

CODE6C = """row = ledger[(ledger.algorithm == "horizontal_flip") & (ledger.parent_id == batch.ids[0])].iloc[0]

x_prime = np.asarray(Image.open(out_root / "payloads" / row.artifact).convert("RGBA"))
undone = x_prime[:, ::-1]        # HWC: the width axis, flipped back

print(f"row: algorithm {row.algorithm!r}, parent {row.parent_id!r}")
print(f"artifact: {row.artifact}")
print(f"undo the flip, compare to the parent: max |Δ| = {np.abs(undone.astype(int) - parent.astype(int)).max()}")
"""

MD7 = """## 7 · A range, by the CLI

One parameter widened into many values: `fraction` as a `{range, samples, mode}` sweep — `mode: even` walks `linspace(0.2, 0.8, 3)`, inclusive ends. The config expands into one transformation instance per value at validation, so the job treats the three fractions as three independent algorithms; two input rows × three fractions = six rows.

The cell builds its own 2-row sidecar exercising the **raw-bytes** image-column form — the third form beside absolute and relative paths: the `img` column holds the RGBA bytes themselves, and `height`/`width` columns carry the shape.
"""

CODE7 = """# --- A range by the CLI: two input frames at three even fractions ---
import pyarrow as pa
import pyarrow.parquet as pq
from kcai_data_sampling_job.cli import run

# Reload the config: this cell must work even if only it was re-run after a YAML edit.
with open("examples/config/walkthrough-procedural.yaml") as f:
    CFG = yaml.safe_load(f)

swp_dir = Path("examples/outputs/sweep")
swp_dir.mkdir(parents=True, exist_ok=True)
i0, i1 = batch.ids[0], batch.ids[1]
table = pa.table({
    "id": [i0, i1],
    "img": [batch.data[0].tobytes(), batch.data[1].tobytes()],
    "height": [batch.data[0].shape[0], batch.data[1].shape[0]],
    "width": [batch.data[0].shape[1], batch.data[1].shape[1]],
})
pq.write_table(table, swp_dir / "samples.parquet")

sweep_cfg = {
    "dataloaders": {"loaders": [{
        "name": "comma10k-sweep", "type": "parquet",
        "path": str(swp_dir / "samples.parquet"),
        "id_column": "id", "load_batch_size": 1, "decode": "img_bytes",
        "sample_path": [{"column": "img"}],                             # the bytes column
    }]},
    "operations": {
        "outputs": {**CFG["operations"]["outputs"], "flush_batch_size": 5},
        "transform_batch_size": 5,          # all three variants of a row in one group
        "transformations": [{
            "name": "crop_resize", "type": "crop_resize",
            "fraction": {"range": [0.2, 0.8], "samples": 3, "mode": "even"},
        }],
    },
}

summary = run(sweep_cfg)                    # CLI-equivalent run()
print("sweep summary:", summary)            # {'comma10k-sweep': 6} = 2×3
"""

MD7B = """The six outputs, three per line — the variants of one input side by side, fractions ascending. The row's `params` carry the resolved `fraction`; `artifact` leads to the pixels.
"""

CODE7B = """import json as _json
from PIL import ImageDraw

swp_out = Path("examples/outputs/comma10k-sweep")
swp_rows = pd.read_parquet(swp_out / "ledger.parquet")
swp_rows["fraction"] = [_json.loads(p)["fraction"] for p in swp_rows["params"]]
swp_rows = swp_rows.sort_values(["parent_id", "fraction"])

def line(label, imgs):                 # one line: label + N variants, side by side
    h, H = max(i.height for i in imgs), max(i.height for i in imgs) + 14
    sheet = Image.new("RGBA", (sum(i.width + 4 for i in imgs) - 4, H), (240, 244, 248, 255))
    x = 0
    for i in imgs:
        sheet.paste(i, (x, 14)); x += i.width + 4
    ImageDraw.Draw(sheet).text((6, 2), label, fill=(60, 80, 100))
    return np.asarray(sheet)

panels = []
for parent, group in swp_rows.groupby("parent_id", sort=True):
    imgs = [Image.open(swp_out / "payloads" / r.artifact).convert("RGBA") for r in group.itertuples()]
    fracs = ", ".join(f"{f:.2g}" for f in group.fraction)
    panels.append((f"input {parent}: {len(group)} outputs", line(f"fractions {fracs}", imgs)))
show(panels, width=1400)                           # three output samples of the same input per line
"""

MD8 = """## 8 · Generative: inpainting with LaMa

A **tool** model produces content, seeded by task knowledge: the region is erased, and LaMa fills it with something that was *not* in the image. The map is deterministic (no seed is drawn) and **not** reversible (what was in the region is gone). One region fill of the 1208×1928 frame takes ~10 s on CPU; running this section needs the opt-in `kcai-data-sampling-lama` install (torch) in the kernel.

The adapter is a registered plugin (`lama_inpaint`, under `kcai_data_sampling.models`); the YAML names it and its `big-lama.pt` weights in `models:`, and the transformation references the **name** — never a Python object. The ledger rows therefore read `algorithm=inpaint`, `tool_model=big-lama`, `family=generative`, `arity=unary`, `reversible=false`, `seed=-`.
"""

CODE8 = """# --- Generative, by the CLI: one region erased and refilled per frame ---
from kcai_data_sampling_job.cli import run

# Reload the config: this cell must work even if only it was re-run after a YAML edit.
with open("examples/config/walkthrough-generative.yaml") as f:
    GEN = yaml.safe_load(f)

# The region the YAML erases (keep in sync with top/left/height/width there).
REGION = {"top": 700, "left": 500, "height": 300, "width": 400}

summary = run(GEN)
print("generative summary:", summary)              # {'comma10k': 10} = 10 frames

gen_root = Path("examples/outputs") / GEN["dataloaders"]["loaders"][0]["name"]
gen_ledger = pd.read_parquet(gen_root / "ledger.parquet")
row = gen_ledger[(gen_ledger.algorithm == "inpaint") & (gen_ledger.parent_id == batch.ids[0])].iloc[0]

original = batch.data[0]
erased = original.copy()
t, l, h, w = REGION["top"], REGION["left"], REGION["height"], REGION["width"]
erased[t:t + h, l:l + w] = 0                       # the erased region, drawn in the notebook
filled = np.asarray(Image.open(gen_root / "payloads" / row.artifact).convert("RGBA"))

show([("original", original), ("region erased", erased), ("inpaint: filled in by LaMa", filled)], width=480)

print("the ledger row that names the fill:", row.algorithm, row.tool_model, row.family,
      row.arity, "reversible =" + str(row.reversible).lower(), "seed =", row.seed)
display(gen_ledger[gen_ledger.parent_id == batch.ids[0]])
"""


MD9 = """## 9 · Adversarial: FGSM against your own model

A **target** model defines the loss, and the transformation steps up its
gradient: one ``epsilon``-sized L∞-bounded step in the direction that **raises**
the loss, in normalized ``[0, 1]`` pixel units (``epsilon * sign(grad)``,
rounded back to the batch dtype). The map is deterministic (no seed) and not
reversible (the pixels moved, they did not flip).

The target model is the **user's**: the YAML's ``models:`` entry names the
adapter **file** (`path:`), and the transformation references the model by
**name** — never a Python object. Running this section needs the opt-in
``kcai-data-sampling-fgsm`` install in the kernel, plus ``torch`` /
``ultralytics`` **and your own weights** for ``grad`` itself (fetched
independently via ``ultralytics``, never by the package); without them every
frame's ``grad`` call fails, and the YAML itself still validates.
"""

CODE9 = """# --- Adversarial, by the CLI: FGSM against the user's target model ---
from kcai_data_sampling_job.cli import run

# Reload the config: this cell must work even if only it was re-run after a YAML edit.
with open("examples/config/walkthrough-adversarial.yaml") as f:
    ADV = yaml.safe_load(f)

summary = run(ADV)
print("adversarial summary:", summary)         # {'comma10k': 10} = 10 frames

adv_root = Path("examples/outputs") / ADV["dataloaders"]["loaders"][0]["name"]
adv_ledger = pd.read_parquet(adv_root / "ledger.parquet")
row = adv_ledger[(adv_ledger.algorithm == "fgsm") & (adv_ledger.parent_id == batch.ids[0])].iloc[0]

original = batch.data[0]
perturbed = np.asarray(Image.open(adv_root / "payloads" / row.artifact).convert("RGBA"))
d = perturbed.astype(np.float64) - original.astype(np.float64)
show([("original", original), (f"fgsm (ε = {json.loads(row.params)['epsilon']:.3g})", perturbed)], width=480)

print("the ledger row that names the attack:", row.algorithm, row.target_model, row.family,
      row.arity, "reversible =" + str(row.reversible).lower(), "seed =", row.seed)
print("max |Δ| over the frame:", float(np.abs(d).max()))
display(adv_ledger[adv_ledger.parent_id == batch.ids[0]])
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
            md_cell(MD7),
            code_cell(CODE7),
            md_cell(MD7B),
            code_cell(CODE7B),
            md_cell(MD8),
            code_cell(CODE8),
            md_cell(MD9),
            code_cell(CODE9),
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
