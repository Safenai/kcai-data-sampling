"""Mirror image arrays horizontally.

Horizontal flip: ``xs[:, :, ::-1, :]`` — the width axis flips, height stays.
"""

from typing import override

import numpy as np

from kcai_data_sampling_core.api.unary import UnaryTransformation
from kcai_data_sampling_images.configs import HorizontalFlipTransformationConfig


class HorizontalFlip(UnaryTransformation):
    """Mirror image arrays horizontally: ``xs[:, :, ::-1, :]``.

    The width axis flips; height stays. The transformation is its own inverse.
    """

    algorithm = "horizontal_flip"

    #: The registered config schema this algorithm validates against.
    Config = HorizontalFlipTransformationConfig

    reversible = True

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, h, w, c) uint8
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Apply the horizontal flip.

        Args:
            xs: Batch of image arrays shaped ``(B, H, W, C)``.
            rngs: Unused for this deterministic operation.

        Returns:
            The flipped batch, contiguous uint8.
        """
        del rngs
        return np.ascontiguousarray(xs[:, :, ::-1, :])