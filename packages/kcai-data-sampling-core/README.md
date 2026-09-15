# `kcai-data-sampling-core`: the sample generation interface

Five transformations (three procedural, one of them n-ary; one generative; one adversarial) and enough of the contract around them to see whether the shape is right. **Nothing in it is a toy**: the dataset is ten comma10k driving frames (MIT), the tool model is LaMa (Apache-2.0), the target model is YOLOv8n, and the tests skip rather than fake a model when its package is not installed.

```bash
pip install -e ".[test,lama,examples]"
python ../../scripts/fetch_comma10k_sample.py    # ten frames + masks, git-ignored
python -m pytest -q                              # 33 tests, ~90 s
```

A worked walkthrough of every claim below, on the real frames, with the images before and after each transformation and every output row shown in full: [`interface_walkthrough.ipynb`](../../examples/notebooks/interface_walkthrough.ipynb), on the comma10k sample, or on WoodScape by changing one line.

## Layout

Mirrors `dqm-ml-core`: a base class in `api/`, one subclass per lifecycle, and the algorithms inheriting from those.

| | Mirrors | |
| :--- | :--- | :--- |
| `api/transformation.py` | `processor.py` | identity, family, reversibility, the declared parameters |
| `api/unary.py` | `features_processor.py` | `T : x ↦ x′`, one parent, `δ` defined; `apply` works on a batch `(B, *sample)` as one array operation |
| `api/n_ary.py` | `metrics_processor.py` | `T : (x₁ … xₙ) ↦ x′`, no `δ`, annotation built |
| `api/selection.py` |, | `Sample`, and the data selection: what a sample is. In memory |
| `api/output.py` |, | `Output`: the bitmap, a reference to the parent, and what produced it. No path |
| `transformations/` | `metrics/` | `HorizontalFlip`, `CropResize`, `CutMix`, `Inpaint`, `FGSM` |
| `api/roles.py` |, | what a model owes for its role: `TargetModel` (`name`, `grad`), `ToolModel` (`name`, `inpaint`); checked at construction and at the first batch |
| `models/` |, | the package's tool model, `LamaTool`, and the weights cache; no target model here, that is the user's (`examples/notebooks/yolo_target.py` is one) |
| `datasets/` |, | one reader per dataset, all exposing the same five names, comma10k (fetched by script) and WoodScape (licensed) |
| `utils/runner.py` | `utils/processor_runner.py` | the runner: a selection in, outputs out; judges nothing, touches no disk; the batch size is chosen there, a memory choice that changes nothing |
| `utils/images.py` |, | the PNG codec the readers use |

The disk is another package, [`kcai-data-sampling-job`](../kcai-data-sampling-job): `Store` puts selections and bitmaps on disk, adds `sample_path` and `data_selection_path` to the rows, and leads back from a row. It depends on this one; this one never imports it.

The interface splits on **arity**, and `CutMix` is the n-ary case: two parents, the right half of the second pasted onto the first.
Two levels decide what goes in, and they must not be confused:
- the **data selection** decides what a *sample* is, four consecutive frames can be one sample,
- while the **algorithm** decides which samples it combines: `CutMix.select_parents` pairs each sample with the next, over the whole selection, before any batching.

## Details

**One signature.** `T : x ↦ x′`. The annotation is not returned for now.

**The family is derived.** `model_role` is `None` / `"tool"` / `"target"`, so procedural / generative / adversarial follows from the role of the model. The row has two slots, `tool_model` and `target_model`, each `NULL` when the algorithm has no model in that role; exactly the slot the role names must be filled, and anything else is refused at construction.

**The target model is the user's; the tool model is the package's.** A role is a contract (`api/roles.py`): a target model is any object with a `name` and the methods the algorithm declares in `model_methods` (`grad` for FGSM), so a user wraps their own network in a few lines and nothing else is imported. What is checkable at construction is checked there (the name, the methods); what only a call reveals (shape, finite values) is checked at the first batch, with the model named in the error. `examples/notebooks/yolo_target.py` is one such wrapper, outside the package, which the notebook and the tests use. Tool models are chosen and shipped by the package (`LamaTool`), under the same kind of contract.

**The parameters are declared.** Each algorithm lists them on the class, `parameters = {"fraction": None, "top": 0, "left": 0}` (`None`: required, otherwise the default), and the constructor resolves the configuration against that list: a missing required one is refused with the list of what is needed and optional, an unknown key is refused rather than recorded, and `params` is always complete, so the row carries every parameter resolved. A model is not a parameter: it is the slot named by `model_role`.

**Inputs and outputs stay in memory; the disk is another package.** A transformation receives `Sample`s and returns `Output`s: the bitmap `x` itself, `parent_id` as the reference to what went in, and the declarations. No path anywhere in the interface, and the runner checks nothing on disk. `kcai_data_sampling_job.Store` is what a campaign calls afterwards, from the job package: it writes the selection once (different content under the same name is refused), each bitmap under a name that carries the selection, the parent, the algorithm, a hash of `params` and `seed` and a hash of the content (so distinct outputs never overwrite each other, two different outputs of one identity land side by side, and a bit-identical re-run is idempotent), and merges the rows into `rows.json`, a ledger. A row on disk is an `Output` minus its bitmap plus the two columns only storage can fill: `sample_path` and `data_selection_path`. `Store.read(row)` and `Store.parent(row)` lead back.

**The seed exists exactly when the algorithm draws.** A deterministic algorithm carries `seed = NULL` (an assertion) and refuses a seed, since two seeds would name two transformations for one output. A stochastic one declares `stochastic = True`, and its seed is then part of its identity. Its draws come from one generator per output, seeded by the seed and the parent's id, so that they depend on neither the batch nor the order of the run.