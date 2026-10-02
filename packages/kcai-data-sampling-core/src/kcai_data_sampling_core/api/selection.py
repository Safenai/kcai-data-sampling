"""What goes into a transformation run: one generic decoded batch.

A **batch is a row**: one ``load_batch_size`` chunk of a selection becomes a
:class:`Batch` — the chunk's source columns stay arrow-native (a pyarrow table,
the sample column dropped) and the sample column decodes **once** into a numpy
``(B, *sample)`` stack. Row ``i`` on every axis is the same sample: ``ids[i]``,
``columns`` slice ``i``, ``data[i]``. There is no ``Sample(x, y)`` and there is
no ``y`` — pyarrow wherever possible, numpy only where the sample math requires
it. Nothing here assumes images: the sample's space is declared by
``sample_axes`` and ``value_range``, both optional because a datatype package
fixes them on its own subclass (``-images`` defines
:class:`~kcai_data_sampling_images.api.selection.ImageBatch`).
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa


@dataclass
class Batch:
    """One decoded chunk of a selection.

    The batch is the in-memory authority on what a sample is, and nothing
    more: how batches relate to each other (streaming order) is a loader
    concern, and how many batches coalesce for execution changes nothing in
    the outputs (batch invariance).

    Attributes:
        name: Unique name of the selection this batch belongs to.
        dataset: Names the reader (loader) that assembled the selection.
        ids: One row identifier per sample, in row order.
        columns: The chunk's source columns as a pyarrow table — the sample
            column is dropped — kept arrow-native for ledger pass-through.
        data: The decoded sample stack ``(B, *sample)``, one row per source
            row, in row order.
        sample_axes: Labels of the sample dimensions, or ``None`` when the
            datatype declares none.
        value_range: ``(low, high)`` domain of sample values, or ``None``.
    """

    name: str
    dataset: str
    ids: list[str]
    columns: pa.Table
    data: np.ndarray
    sample_axes: tuple[str, ...] | None = None
    value_range: tuple[float, float] | None = None

    def __len__(self) -> int:
        """Return the number of rows (samples) in this batch."""
        return len(self.data)

    def row(self, index: int) -> "Batch":
        """Return a single-row batch: the sample at ``index``.

        Slicing keeps the pyarrow columns and the numpy stack aligned with the
        ids, so a row is usable anywhere the whole batch is (the runner, a
        unary ``transform``), at ``(B=1, *sample)``.

        Args:
            index: Row index into every axis of the batch.

        Returns:
            The one-row batch.
        """
        return Batch(
            name=self.name,
            dataset=self.dataset,
            ids=[self.ids[index]],
            columns=self.columns.slice(index, 1),
            data=self.data[index : index + 1],
            sample_axes=self.sample_axes,
            value_range=self.value_range,
        )
