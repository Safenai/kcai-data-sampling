# kcai-workspace

Packages for the KCAI data sampling work.

| Package | |
| :--- | :--- |
| [`kcai-data-sampling`](packages/kcai-data-sampling) | the sample generation interface: transformations, their families, what one output row carries, and what it leaves to the modules downstream |

## Run it

```bash
pip install -e "packages/kcai-data-sampling[test,yolo,lama,examples]"
python scripts/fetch_comma10k_sample.py               # ten frames + masks, ~20 MB, git-ignored
python -m pytest packages/kcai-data-sampling -q       # 34 tests, ~80 s
jupyter lab examples/notebooks/interface_walkthrough.ipynb
```

The data is ten [comma10k](https://github.com/commaai/comma10k) driving frames
with their masks, MIT-licensed, fetched into `examples/data/comma10k_sample/`.
The models' weights (YOLOv8n 6 MB, LaMa 206 MB) download once, into
`~/.cache/kcai-data-sampling/weights/` (override with `$KCAI_WEIGHTS_DIR`).

## Examples

| | |
| :--- | :--- |
| [`examples/notebooks/interface_walkthrough.ipynb`](examples/notebooks/interface_walkthrough.ipynb) | the interface end to end on real driving frames, LaMa and YOLOv8n, images before and after each transformation, a region inpainted, a real detector attacked, every output row shown in full, and where the interface stops. Runs on the comma10k sample; **one line switches it to WoodScape** |
| [`examples/notebooks/display.py`](examples/notebooks/display.py) | the notebook's display helpers, images side by side, detections drawn on, rows as a full table. Presentation only, nothing of the interface |
| [`scripts/fetch_comma10k_sample.py`](scripts/fetch_comma10k_sample.py) | downloads the ten frames and their masks into `examples/data/comma10k_sample/` |

**WoodScape**, Valeo's fisheye corpus, is also supported (`kcai_data_sampling.datasets.woodscape`)
but not committed: its data is under a proprietary licence. Whoever has accepted
it drops the files into `examples/data/woodscape_sample/`, which is git-ignored,
see the README there, and flips the one commented line in the notebook's setup
cell. Every dataset reader exposes the same five names (`samples`, `load_sample`, `describe`,
`overlay`, `ground_truth`), which is what makes
that a one-line change.
