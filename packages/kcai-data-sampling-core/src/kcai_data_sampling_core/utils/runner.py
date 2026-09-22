"""The runner: a selection in, outputs out. Nothing touches the disk.

Genericity rule: batches are execution details only — any split of the
selection gives the same outputs in the same order (batch invariance), which is
the license for the phase-9 parallel runner.
"""

from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_core.api.selection import DataSelection, Sample
from kcai_data_sampling_core.api.transformation import Transformation


class TransformationRunner:
    """Runs one transformation over an in-memory selection (or a subset).

    Attributes:
        selection: The in-memory selection the runner transforms.
    """

    def __init__(self, selection: DataSelection):
        """Build the runner for a selection.

        Args:
            selection: The in-memory selection; its ``value_range`` is handed
                to every transformation call.
        """
        self.selection = selection

    def run(
        self,
        transformation: Transformation,
        samples: list[Sample] | None = None,
        batch_size: int | None = None,
    ) -> list[Output]:
        """Apply one transformation to the selection (or a subset of it).

        ``batch_size`` is an execution detail: any split gives the same outputs
        in the same order (unary partition invariance; n-ary parent sets are
        formed once over the whole set, then split, so pairing never depends on
        batching).

        Args:
            transformation: The transformation to apply.
            samples: The samples to transform; ``None`` uses the whole
                selection.
            batch_size: Target ``(B, *sample)`` batch handed to ``apply``;
                ``None`` transforms the whole set in one batch.

        Returns:
            One ``Output`` per transformed sample, in selection order.
        """
        samples = self.selection.samples if samples is None else samples
        units = transformation.select_parents(samples) if transformation.arity == "n-ary" else samples
        size = len(units) if batch_size is None else batch_size
        return [
            out
            for start in range(0, len(units), max(size, 1))
            for out in transformation.transform(units[start : start + size], self.selection.value_range)
        ]