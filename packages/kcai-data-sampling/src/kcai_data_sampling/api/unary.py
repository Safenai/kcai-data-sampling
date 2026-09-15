"""Unary transformation base class.

    T : x ↦ x′

The algorithm writes one method, ``apply``. ``x′`` lives in the same space as
``x``: a contract checked on every output, which is what keeps ``δ = x′ − x``
defined. An algorithm that would shrink the sample resamples back; one that
would enlarge it is out of scope.
"""

import numpy as np

from kcai_data_sampling.api.record import Record
from kcai_data_sampling.api.selection import Sample
from kcai_data_sampling.api.transformation import Transformation


class UnaryTransformation(Transformation):
    arity = "unary"

    def apply(self, x: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
        """Produce x′ from x, in the same space. `rng` is None when deterministic."""
        raise NotImplementedError

    def transform(self, sample: Sample) -> tuple[np.ndarray, Record]:
        """One sample → ``(x_prime, record)``. The annotation is not touched."""
        x_prime = np.clip(self.apply(sample.x, self.rng()), 0.0, 1.0)
        if x_prime.shape != sample.x.shape:
            raise ValueError(
                f"{self.algorithm} changed the sample space: {sample.x.shape} → {x_prime.shape}. "
                "A unary transformation preserves it, so that δ stays defined."
            )
        return x_prime, Record(data_selection_path=None, parent_id=sample.id, **self.describe())

    def transform_batch(self, batch: list[Sample]) -> list[tuple[np.ndarray, Record]]:
        """A loop: the batch carries no meaning. A vectorising backend that
        overrides this owes the same rows."""
        return [self.transform(sample) for sample in batch]
