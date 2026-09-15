# `kcai-data-sampling-core`: the sample generation interface

Five transformations (three procedural, one of them n-ary; one generative; one adversarial) and enough of the contract around them to see whether the shape is right. **Nothing in it is a toy**: the dataset is ten comma10k driving frames (MIT), the tool model is LaMa (Apache-2.0), the target model is YOLOv8n, and the tests skip rather than fake a model when its package is not installed.

```bash
pip install -e ".[test,lama,examples]"
python ../../scripts/fetch_comma10k_sample.py    # ten frames + masks, git-ignored
python -m pytest -q                              # 42 tests, ~2 min
```

A worked walkthrough of every claim below, on the real frames, with the images before and after each transformation and every output row shown in full: [`interface_walkthrough.ipynb`](../../examples/notebooks/interface_walkthrough.ipynb), on the comma10k sample, or on WoodScape by changing one line.

## Layout

Mirrors `dqm-ml-core`: a base class in `api/`, one subclass per lifecycle, and the algorithms inheriting from those.

| | Mirrors | |
| :--- | :--- | :--- |
| `api/transformation.py` | `processor.py` | identity, family, reversibility, the declared parameters |
| `api/unary.py` | `features_processor.py` | `T : x ↦ x′`, one parent, `δ` defined; `apply` works on a batch `(B, *sample)` as one array operation |
| `api/n_ary.py` | `metrics_processor.py` | `T : (x₁ … xₙ) ↦ x′`, no `δ`, annotation built |
| `api/selection.py` |, | `Sample`, and the data selection: what a sample is, written once before the run |
| `api/record.py` |, | what one output row carries, and does not |
| `transformations/` | `metrics/` | `HorizontalFlip`, `CropResize`, `CutMix`, `Inpaint`, `FGSM` |
| `api/roles.py` |, | what a model owes for its role: `TargetModel` (`name`, `grad`), `ToolModel` (`name`, `inpaint`); checked at construction and at the first batch |
| `models/` |, | the package's tool model, `LamaTool`, and the weights cache; no target model here, that is the user's (`examples/notebooks/yolo_target.py` is one) |
| `datasets/` |, | one reader per dataset, all exposing the same five names, comma10k (fetched by script) and WoodScape (licensed) |
| `utils/runner.py` | `utils/processor_runner.py` | the runner: checks, stamps; judges nothing; the batch size is chosen there, a memory choice that changes nothing |

The interface splits on **arity**, and `CutMix` is the n-ary case: two parents, the right half of the second pasted onto the first.
Two levels decide what goes in, and they must not be confused:
- the **data selection** decides what a *sample* is, four consecutive frames can be one sample,
- while the **algorithm** decides which samples it combines: `CutMix.select_parents` pairs each sample with the next, over the whole selection, before any batching.

## Details

**One signature.** `T : x ↦ x′`. The annotation is not returned for now.

**The family is derived.** `model_role` is `None` / `"tool"` / `"target"`, so procedural / generative / adversarial follows from the role of the model. The row has two slots, `tool_model` and `target_model`, each `NULL` when the algorithm has no model in that role; exactly the slot the role names must be filled, and anything else is refused at construction.

**The target model is the user's; the tool model is the package's.** A role is a contract (`api/roles.py`): a target model is any object with a `name` and the methods the algorithm declares in `requires` (`grad` for FGSM), so a user wraps their own network in a few lines and nothing else is imported. What is checkable at construction is checked there (the name, the methods); what only a call reveals (shape, finite values) is checked at the first batch, with the model named in the error. `examples/notebooks/yolo_target.py` is one such wrapper, outside the package, which the notebook and the tests use. Tool models are chosen and shipped by the package (`LamaTool`), under the same kind of contract.

**The parameters are declared.** Each algorithm lists them on the class, `parameters = {"fraction": None, "top": 0, "left": 0}` (`None`: required, otherwise the default), and the constructor resolves the configuration against that list: a missing required one is refused with the list of what is needed and optional, an unknown key is refused rather than recorded, and `params` is always complete, so the row carries every parameter resolved. A model is not a parameter: it is the slot named by `model_role`.

**The seed exists exactly when the algorithm draws.** A deterministic algorithm carries `seed = NULL` (an assertion) and refuses a seed, since two seeds would name two transformations for one output. A stochastic one declares `stochastic = True`, and its seed is then part of its identity. Its draws come from one generator per output, seeded by the seed and the parent's id, so that they depend on neither the batch nor the order of the run.