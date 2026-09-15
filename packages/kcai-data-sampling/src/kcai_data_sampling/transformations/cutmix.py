"""CutMix, basic: n-ary, procedural.

Two parents; the right part of the second is pasted onto the first. Pairing is
cyclic over the selection (each sample with the next), so it draws nothing.
"""

import numpy as np
from typing_extensions import override

from kcai_data_sampling.api.n_ary import NAryTransformation
from kcai_data_sampling.api.selection import Sample


class CutMix(NAryTransformation):
    """Params: ``fraction`` of the width taken from the second parent (default 0.5)."""

    algorithm = "cutmix"
    reversible = False  # part of each parent is gone

    @override
    def select_parents(self, samples: list[Sample]) -> list[tuple[list[Sample], list[float]]]:
        fraction = self.params.get("fraction", 0.5)
        return [([s, samples[(i + 1) % len(samples)]], [1 - fraction, fraction]) for i, s in enumerate(samples)]

    @override
    def combine(self, xs: list[np.ndarray], weights: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        x1, x2 = xs
        width = x1.shape[-1]
        right = np.arange(width) >= np.round(width * (1 - weights[:, 1]))[:, None]  # (B, W): from the second parent
        return np.where(right[:, None, None, :], x2, x1)
