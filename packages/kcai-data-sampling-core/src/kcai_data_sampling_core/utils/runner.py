"""The runner: a batch in, outputs out. Nothing touches the disk.

Genericity rule: batches are execution details only — any split of a selection
gives the same outputs in the same order (batch invariance), which is what
lets a parallel runner arrive without changing the row contract.
"""

from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_core.api.selection import Batch
from kcai_data_sampling_core.api.transformation import Transformation


class TransformationRunner:
    """Runs one transformation over an in-memory batch (or a subset).

    Attributes:
        batch: The in-memory batch the runner transforms.
    """

    def __init__(self, batch: Batch):
        """Build the runner for a batch.

        Args:
            batch: The in-memory batch; its ``value_range`` is handed to every
                transformation call.
        """
        self.batch = batch

    def run(
        self,
        transformation: Transformation,
        batch: Batch | None = None,
    ) -> list[Output]:
        """Apply one transformation to the batch (or a subset of it).

        Args:
            transformation: The transformation to apply.
            batch: The batch to transform; ``None`` uses the runner's own
                batch.

        Returns:
            One ``Output`` per transformed row, in batch order.
        """
        batch = self.batch if batch is None else batch
        units = transformation.select_parents(batch) if transformation.arity == "n-ary" else batch
        return transformation.transform(units, batch.value_range)