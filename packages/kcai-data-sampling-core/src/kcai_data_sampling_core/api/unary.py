"""Unary transformation base class.

    T : x ↦ x′

The algorithm writes one method, ``apply``, on a **batch**: an array of shape
``(B, *sample)`` in, the same shape out, computed as one array operation.
``x′`` lives in the same space as ``x`` — its shape and its range of values —
a contract that is what keeps ``δ = x′ − x`` defined. The shape is checked
here. The range is the selection's, handed to ``transform`` by the runner: an
algorithm that declares ``clips = True`` has its output clipped to it, any
other output outside it is refused. An algorithm that would shrink the sample
resamples back; one that would enlarge it is out of scope.
"""

import numpy as np

from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_core.api.selection import Sample
from kcai_data_sampling_core.api.transformation import Transformation


class UnaryTransformation(Transformation):
    """Arity-fixed base for one-input-one-output transformations.

    Arity is fixed here: a unary transformation maps ``(B, *sample)`` to
    ``(B, *sample)``, one output per input sample.
    """

    arity = "unary"

    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        """Map ``(B, *sample)`` → ``(B, *sample)``, in the same space.

        Args:
            xs: Input batch ``(B, *sample)``.
            rngs: One generator per row when the algorithm draws randomness,
                ``None`` when deterministic.

        Returns:
            The transformed batch, same shape as ``xs``.

        Raises:
            NotImplementedError: Subclasses must implement ``apply``.
        """
        raise NotImplementedError

    def transform(self, batch: list[Sample], value_range: tuple[float, float] | None = None) -> list[Output]:
        """Transform one batch into one ``Output`` per sample.

        Args:
            batch: The input samples of one batch.
            value_range: The selection's ``(low, high)``, or ``None`` to skip
                the clip and the range check.

        Returns:
            One ``Output`` per sample, in batch order.

        Raises:
            ValueError: If ``apply`` changed the sample space, or the output
                left the declared value range.
        """
        xs = np.stack([s.x for s in batch])
        out = self.apply(xs, self.rngs([s.id for s in batch]))
        if out.shape != xs.shape:
            raise ValueError(
                f"{self.algorithm} changed the sample space: {xs.shape[1:]} → {out.shape[1:]}. "
                "A unary transformation preserves it, so that δ stays defined."
            )
        out = self.fit_to_range(out, value_range)
        return [Output(x=x_prime, parent_id=s.id, **self.describe()) for x_prime, s in zip(out, batch)]