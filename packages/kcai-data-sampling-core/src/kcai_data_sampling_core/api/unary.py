"""Unary transformation base class.

    T : x ↦ x′

The algorithm writes one method, ``apply``, on a **batch**: an array of
shape ``(B, *sample)`` in, the same shape out, computed as one array
operation. ``x′`` lives in the same space as ``x``: a contract checked on
every output, which is what keeps ``δ = x′ − x`` defined. An algorithm that
would shrink the sample resamples back; one that would enlarge it is out of
scope.
"""

import numpy as np

from kcai_data_sampling_core.api.record import Record
from kcai_data_sampling_core.api.selection import Sample
from kcai_data_sampling_core.api.transformation import Transformation


class UnaryTransformation(Transformation):
    arity = "unary"

    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        """``(B, *sample)`` → ``(B, *sample)``, in the same space. ``rngs`` is one
        generator per row, ``None`` when deterministic."""
        raise NotImplementedError

    def transform_batch(self, batch: list[Sample]) -> list[tuple[np.ndarray, Record]]:
        """One batch → one ``(x_prime, record)`` per sample. The annotation is not touched."""
        xs = np.stack([s.x for s in batch])
        out = np.clip(self.apply(xs, self.rngs([s.id for s in batch])), 0.0, 1.0)
        if out.shape != xs.shape:
            raise ValueError(
                f"{self.algorithm} changed the sample space: {xs.shape[1:]} → {out.shape[1:]}. "
                "A unary transformation preserves it, so that δ stays defined."
            )
        return [
            (x_prime, Record(data_selection_path=None, parent_id=s.id, **self.describe()))
            for x_prime, s in zip(out, batch)
        ]

    def transform(self, sample: Sample) -> tuple[np.ndarray, Record]:
        """One sample: a batch of one."""
        return self.transform_batch([sample])[0]
