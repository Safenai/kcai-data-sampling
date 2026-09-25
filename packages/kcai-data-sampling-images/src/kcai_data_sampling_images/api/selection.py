"""What the image packages feed the generic engine: one decoded image batch.

:class:`ImageBatch` is the image specialization of the generic core
:class:`~kcai_data_sampling_core.api.selection.Batch`: the sample's space is
fixed here (``("height", "width", "channel")``, ``(0, 255)``) and the decoded
stack is exposed as ``images`` for the image math. Row ``i`` on every axis is
the same sample: ``ids[i]``, ``columns`` slice ``i``, ``images[i]``.
"""

from dataclasses import dataclass

import numpy as np

from kcai_data_sampling_core.api.selection import Batch


@dataclass
class ImageBatch(Batch):
    """One decoded image chunk: the image specialization of :class:`Batch`.

    Attributes:
        name: Unique name of the selection this batch belongs to.
        dataset: Names the reader (loader) that assembled the selection.
        ids: One row identifier per image, in row order.
        columns: The chunk's source columns as a pyarrow table — the image
            column is dropped — kept arrow-native for ledger pass-through.
        data: The decoded image stack ``(B, H, W, 4)`` uint8, one row per
            source row, in row order.
        sample_axes: Labels of the image dimensions ``(H, W, C)``.
        value_range: ``(low, high)`` domain of image values.
    """

    sample_axes: tuple[str, ...] = ("height", "width", "channel")
    value_range: tuple[float, float] = (0.0, 255.0)

    @property
    def images(self) -> np.ndarray:
        """The decoded image stack, an alias of ``data`` for the image math."""
        return self.data

    def row(self, index: int) -> "ImageBatch":
        """Return a single-row batch: the image at ``index``.

        Args:
            index: Row index into every axis of the batch.

        Returns:
            The one-row batch.
        """
        return ImageBatch(
            name=self.name,
            dataset=self.dataset,
            ids=[self.ids[index]],
            columns=self.columns.slice(index, 1),
            data=self.data[index : index + 1],
            sample_axes=self.sample_axes,
            value_range=self.value_range,
        )