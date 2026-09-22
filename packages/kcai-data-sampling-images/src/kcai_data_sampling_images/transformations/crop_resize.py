"""Crop a central window, then resize it back to the sample shape.

Crop-resize: keep the ``top:top+fraction*H`` by ``left:left+fraction*W``
window, then nearest-neighbor resize back to ``(H, W)`` by index-replication.
"""

from typing import Any, override

import numpy as np

from kcai_data_sampling_core.api.unary import UnaryTransformation
from kcai_data_sampling_images.configs import CropResizeTransformationConfig


class CropResize(UnaryTransformation):
    """Crop a fraction of the image (top-left anchored), then resize it back.

    ``fraction`` is required (its value ``None`` marks it); ``top`` and
    ``left`` default to 0. The crop discards information (not reversible), and
    the resize back to the original shape keeps the sample's space intact.
    """

    algorithm = "crop_resize"

    #: The registered config schema this algorithm validates against.
    Config = CropResizeTransformationConfig

    parameters: dict[str, Any] = {
        "fraction": None,
        "top": 0,
        "left": 0,
    }

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
        height, width = xs.shape[1], xs.shape[2]
        window_h: int = max(1, int(height * self.params["fraction"]))
        window_w: int = max(1, int(width * self.params["fraction"]))
        top: int = self.params["top"]
        left: int = self.params["left"]
        if top + window_h > height or left + window_w > width:
            raise ValueError(
                f"{self.algorithm}: crop window {window_h}x{window_w} at ({top},{left}) "
                f"leaves the {height}x{width} image"
            )
        window = xs[:, top : top + window_h, left : left + window_w, :]

        rows: np.ndarray = np.arange(height) * window_h // height
        cols: np.ndarray = np.arange(width) * window_w // width
        return window[:, rows[:, None], cols[None, :], :]