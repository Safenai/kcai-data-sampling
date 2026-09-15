"""Dataset readers. A reader knows one dataset's layout and produces
`Sample` objects; the annotation travels through the interface untouched.

Every reader exposes the same five names, so the notebook switches dataset
in one line:

| name | |
| --- | --- |
| `samples(root, limit=None)` | samples read out of the dataset, each with its `source` |
| `load_sample(id, source)` | one sample assembled from what a selection recorded, how a row is followed back |
| `describe(samples)` | what was found, printed |
| `overlay(x, y)` | the annotation drawn on the image, for display |
| `ground_truth(sample)` | a one-line summary of the annotation |
"""

from kcai_data_sampling_core.datasets import comma10k, woodscape

__all__ = ["comma10k", "woodscape"]
