"""The ``crop_resize`` transformation class.

Wraps the crop-and-resize-back math
(:func:`kcai_data_sampling_images.transformations.crop_resize.crop_resize`)
into the unary transformation contract.
"""

from typing import override

from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np

from kcai_data_sampling_images.configs import CropResizeTransformationConfig
from kcai_data_sampling_images.transformations.crop_resize import crop_resize


class CropResize(UnaryTransformation):
    """Crop a fraction of the image (top-left anchored), then resize it back.

    ``fraction`` is required (its schema marks it so); ``top`` and ``left``
    default to 0. The crop discards information (not reversible), and the
    resize back to the original shape keeps the sample's space intact.
    """

    algorithm = "crop_resize"

    #: The registered config schema this algorithm validates against.
    Config = CropResizeTransformationConfig

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, h, w, c) uint8
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Crop then resize the batch.

        Args:
            xs: Batch of image arrays shaped ``(B, H, W, C)``.
            rngs: Unused for this deterministic operation.

        Returns:
            A batch of the same shape, uint8.

        Raises:
            ValueError: If the crop window leaves the image (``top``/``left``
                too large for the computed window).
        """
        del rngs
        return crop_resize(
            xs,
            fraction=self.params["fraction"],
            top=self.params["top"],
            left=self.params["left"],
        )
