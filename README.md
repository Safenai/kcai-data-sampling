# kcai-workspace

Packages for the KCAI data sampling work.

Laid out like `dqm-ml-workspace`: a core that works in memory, and a job package that owns the disk and depends on the core, never the reverse.

| Package | |
| :--- | :--- |
| [`kcai-data-sampling-core`](packages/kcai-data-sampling-core) | the sample generation interface: transformations, their families, what one output carries, and what it leaves to the modules downstream. Nothing in it touches the disk |
| [`kcai-data-sampling-job`](packages/kcai-data-sampling-job) | `Store`: writes selections and outputs, adds the two path columns, leads back from a row |

## Run it

```bash
pip install -e "packages/kcai-data-sampling-core[test,lama,examples]" -e "packages/kcai-data-sampling-job[test]"
python scripts/fetch_comma10k_sample.py               # ten frames + masks, ~20 MB, git-ignored
python -m pytest packages/kcai-data-sampling-core -q       # 35 tests, ~2 min
python -m pytest packages/kcai-data-sampling-job -q        # 8 tests
jupyter lab examples/notebooks/interface_walkthrough.ipynb
```

The data is ten [comma10k](https://github.com/commaai/comma10k) driving frames
with their masks, MIT-licensed, fetched into `examples/data/comma10k_sample/`.
The models' weights (YOLOv8n 6 MB, LaMa 206 MB) download once, into
`~/.cache/kcai-data-sampling-core/weights/` (override with `$KCAI_WEIGHTS_DIR`).

## Examples

| | |
| :--- | :--- |
| [`examples/notebooks/interface_walkthrough.ipynb`](examples/notebooks/interface_walkthrough.ipynb) | the interface end to end on real driving frames, LaMa and YOLOv8n, images before and after each transformation, a region inpainted, a real detector attacked, every output row shown in full, and where the interface stops. Runs on the comma10k sample; **one line switches it to WoodScape** |
| [`examples/notebooks/display.py`](examples/notebooks/display.py) | the notebook's display helpers, images side by side, detections drawn on, rows as a full table. Presentation only, nothing of the interface |
| [`examples/notebooks/yolo_target.py`](examples/notebooks/yolo_target.py) | YOLOv8n wrapped as a target model, the way a user wraps their own: `name` and `grad`. Outside the package on purpose |
| [`scripts/fetch_comma10k_sample.py`](scripts/fetch_comma10k_sample.py) | downloads the ten frames and their masks into `examples/data/comma10k_sample/` |

**WoodScape**, Valeo's fisheye corpus, is also supported (`kcai_data_sampling_core.datasets.woodscape`)
but not committed: its data is under a proprietary licence. Whoever has accepted
it drops the files into `examples/data/woodscape_sample/`, which is git-ignored,
see the README there, and flips the one commented line in the notebook's setup
cell. Every dataset reader exposes the same five names (`samples`, `load_sample`, `describe`,
`overlay`, `ground_truth`), which is what makes
that a one-line change.
