"""The runner: a selection in, outputs out. Nothing touches the disk."""

from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_core.api.selection import DataSelection, Sample
from kcai_data_sampling_core.api.transformation import Transformation


class TransformationRunner:
    def __init__(self, selection: DataSelection):
        self.selection = selection

    def run(
        self,
        transformation: Transformation,
        samples: list[Sample] | None = None,
        batch_size: int | None = None,
    ) -> list[Output]:
        """Apply one transformation to the selection (or a subset of it).

        ``batch_size`` is an execution detail: any split gives the same
        outputs in the same order. For an n-ary transformation the parent
        sets are formed once over every sample, then split. The selection's
        ``value_range`` is handed to every call.
        """
        samples = self.selection.samples if samples is None else samples
        units = transformation.select_parents(samples) if transformation.arity == "n-ary" else samples
        size = len(units) if batch_size is None else batch_size
        return [
            out
            for start in range(0, len(units), max(size, 1))
            for out in transformation.transform(units[start : start + size], self.selection.value_range)
        ]
