"""N-ary transformation base class.

    T : (x₁ … xₙ) ↦ x′

The interface splits on arity because arity changes the contract: ``δ`` is
undefined (no single source), the annotation is constructed rather than
inherited, and lineage is ``n`` parents with weights.

Two levels decide what goes in. The data selection decides what a *sample*
is; the algorithm decides which samples it combines, otherwise everything
would be unary. So ``select_parents`` sees the whole selection, never a
batch: pairing inside a batch would make the pairs depend on the batch size.
``apply`` then works on a batch of parent sets, one array per parent slot.
"""

import numpy as np

from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_core.api.selection import Sample
from kcai_data_sampling_core.api.transformation import Transformation


class NAryTransformation(Transformation):
    arity = "n-ary"

    def select_parents(self, samples: list[Sample]) -> list[tuple[list[Sample], list[float]]]:
        """Called once over every sample of the selection. Returns one
        ``(parents, weights)`` per output, every parent set of the same size.
        A random pairing is a draw: such an algorithm is ``stochastic`` and
        its seed fixes the pairs."""
        raise NotImplementedError

    def apply(self, xs: list[np.ndarray], weights: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        """``xs[k]`` is the ``k``-th parent of every output, ``(B, *sample)``;
        ``weights`` is ``(B, n)``. Returns ``(B, *sample)``."""
        raise NotImplementedError

    def transform(
        self, batch: list[tuple[list[Sample], list[float]]], value_range: tuple[float, float] | None = None
    ) -> list[Output]:
        """One batch of parent sets, already chosen over the whole selection.
        ``value_range`` as for the unary case."""
        n = len(batch[0][0])
        xs = [np.stack([parents[k].x for parents, _ in batch]) for k in range(n)]
        weights = np.asarray([w for _, w in batch], dtype="float32")
        keys = [",".join(p.id for p in parents) for parents, _ in batch]
        out = self.fit_to_range(self.apply(xs, weights, self.rngs(keys)), value_range)
        return [Output(x=x_prime, parent_id=key, **self.describe()) for x_prime, key in zip(out, keys)]
