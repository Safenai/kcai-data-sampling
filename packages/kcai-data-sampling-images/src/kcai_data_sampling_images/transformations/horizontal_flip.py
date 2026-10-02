"""Mirror image arrays horizontally: the algorithm math.

``horizontal_flip(xs)`` is the pure numpy map; the transformation class that
wraps it into the unary contract lives in
``kcai_data_sampling_images.api.transformations.horizontal_flip``.
"""

import numpy as np


def horizontal_flip(xs: np.ndarray) -> np.ndarray:
    """Mirror a batch of image arrays on the width axis.

    ``xs[:, :, ::-1, :]`` — the width axis flips, height stays. The map is its
    own inverse.

    Args:
        xs: Batch of image arrays shaped ``(B, H, W, C)``.

    Returns:
        The flipped batch, contiguous uint8.
    """
    return np.ascontiguousarray(xs[:, :, ::-1, :])
