# `kcai-data-sampling-job`: the disk

The interface (`kcai-data-sampling-core`) works in memory: a transformation receives `Sample`s and returns `Output`s, the bitmap and a reference to its parent, and knows no path. This package is what a campaign calls afterwards.

```python
from kcai_data_sampling_job import Store

store = Store("outputs")                      # selections/ and transformed_samples/ under it
rows = store.write(selection, outputs)        # the selection once, one PNG per output, rows.json merged
x_prime = store.read(rows[0])                 # the bitmap back
parent = store.parent(rows[0])                # the selection file, the entry by id, the reader it names
```

**What it guarantees.** The selection is written once: same content again is idempotent, different content under the same name is refused, since rows produced against the first say what it was. A file name carries the selection, the parent, the algorithm, a hash of `params` and `seed` and a hash of the content as written: distinct outputs never overwrite each other, two different outputs of one identity (non-deterministic kernels) land side by side, a bit-identical re-run lands on the same name. `rows.json` is a ledger of every output in the directory, across runs and campaigns. Paths are stored as given.

**What it adds to a row.** Exactly two columns, `data_selection_path` and `sample_path`, because only it knows them. Everything else on a row is the `Output` as the interface produced it, minus the bitmap.

```bash
pip install -e "../kcai-data-sampling-core[test,lama,examples]" -e ".[test]"
python -m pytest -q        # 8 tests
```
