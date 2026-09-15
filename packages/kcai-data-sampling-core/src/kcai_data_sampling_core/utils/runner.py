"""The runner that fills the rows."""

import numpy as np

from kcai_data_sampling_core.api.record import Record
from kcai_data_sampling_core.api.selection import DataSelection, Sample
from kcai_data_sampling_core.api.transformation import Transformation


class TransformationRunner:
    """Run transformations over a saved data selection; stamp each row with it."""

    def __init__(self, selection: DataSelection):
        self.selection = selection

    def run(
        self,
        transformation: Transformation,
        samples: list[Sample] | None = None,
        batch_size: int | None = None,
    ) -> list[tuple[np.ndarray, Record]]:
        """Apply one transformation to the selection (or a subset of it).

        ``batch_size`` is an execution detail: any split gives the same rows in
        the same order. For an n-ary transformation the parent sets are
        formed once over every sample, then split.
        """
        if self.selection.path is None:
            raise ValueError(
                "the data selection has not been saved: a row references its selection by "
                "path, call selection.save(directory) first"
            )

        samples = self.selection.samples if samples is None else samples
        units = transformation.select_parents(samples) if transformation.arity == "n-ary" else samples
        size = len(units) if batch_size is None else batch_size
        results = [
            out
            for start in range(0, len(units), max(size, 1))
            for out in transformation.transform_batch(units[start : start + size])
        ]

        for _, record in results:
            record.data_selection_path = self.selection.path
        return results
