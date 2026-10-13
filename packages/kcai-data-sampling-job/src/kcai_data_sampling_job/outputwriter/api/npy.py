"""The ``npy`` payload writer: one hashed ``.npy`` file per row.

The payload format for samples that are not 8-bit images — time-series
windows ``(W, C)`` float, or any other array: ``np.save`` keeps the dtype and
the shape exactly, with no lossy cast. Buffering, flushing, trace-only mode and
content-addressed names are the ``images`` writer's; only the file format
differs.
"""

import io
from typing import ClassVar

import numpy as np
from typing_extensions import override

from kcai_data_sampling_job.outputwriter.api.images import ImagesOutputWriter


class NpyOutputWriter(ImagesOutputWriter):
    """Writes one ``.npy`` file per output row.

    The plugin for the ``npy`` registered type
    (``kcai_data_sampling.outputwriter`` entry point), chosen with
    ``outputs.payload_format: npy``.

    Attributes:
        suffix: ``npy``.
    """

    suffix: ClassVar[str] = "npy"

    @override
    def encode(self, x: np.ndarray) -> bytes:
        """Serialize one sample with ``np.save`` (no pickle).

        Args:
            x: The sample array, any shape and numeric dtype.

        Returns:
            The ``.npy`` file bytes; ``np.load`` gives back ``x`` exactly.
        """
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(x), allow_pickle=False)
        return buffer.getvalue()
