"""The ``horizontal_flip`` transformation class.

Wraps the mirrored-width-axis math
(:func:`kcai_data_sampling_images.transformations.horizontal_flip.horizontal_flip`)
into the unary transformation contract.
"""

from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_images.configs import HorizontalFlipTransformationConfig
from kcai_data_sampling_images.transformations.horizontal_flip import horizontal_flip


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
        return horizontal_flip(xs)
