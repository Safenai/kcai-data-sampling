"""N-ary transformation base class.

    T : (x₁ … xₙ) ↦ x′

The interface splits on arity because
arity changes the contract: ``δ`` is undefined (no single source), the
annotation is constructed rather than inherited, and lineage is ``n`` parents
with weights.

Two levels decide what goes in. The data selection decides what a *sample*
is; the algorithm decides which samples it combines, otherwise everything
would be unary. So ``select_parents`` sees the whole selection, never a
batch: pairing inside a batch would make the pairs depend on the batch size.
"""

import numpy as np

from kcai_data_sampling.api.record import Record
from kcai_data_sampling.api.selection import Sample
from kcai_data_sampling.api.transformation import Transformation


class NAryTransformation(Transformation):
    arity = "n-ary"

    def select_parents(self, samples: list[Sample]) -> list[tuple[list[Sample], list[float]]]:
        """Called once over every sample of the selection. Returns one
        ``(parents, weights)`` per output. A random pairing is a draw: such an
        algorithm is ``stochastic`` and its seed fixes the pairs."""
        raise NotImplementedError

    def combine(self, xs: list[np.ndarray], weights: list[float], rng: np.random.Generator | None) -> np.ndarray:
        """Produce x′ from the parents' data."""
        raise NotImplementedError

    def transform(self, parents: list[Sample], weights: list[float]) -> tuple[np.ndarray, Record]:
        x_prime = np.clip(self.combine([p.x for p in parents], weights, self.rng()), 0.0, 1.0)
        return x_prime, Record(
            data_selection_path=None, parent_id=",".join(p.id for p in parents), **self.describe()
        )

    def transform_batch(self, batch: list[tuple[list[Sample], list[float]]]) -> list[tuple[np.ndarray, Record]]:
        """A loop over parent sets already chosen over the whole selection."""
        return [self.transform(parents, weights) for parents, weights in batch]
