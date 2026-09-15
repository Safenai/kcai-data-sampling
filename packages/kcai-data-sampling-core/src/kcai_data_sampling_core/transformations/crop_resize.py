"""Crop and resize, procedural.

Keep a window, resample it back to the original size: what keeps ``x′`` in
the same space as ``x`` (a bare crop would be refused), and what the
robustness literature does when it stresses resolution.
"""

import numpy as np
from typing_extensions import override

from kcai_data_sampling_core.api.unary import UnaryTransformation


class CropResize(UnaryTransformation):
    """``fraction`` of each side kept, from ``top`` / ``left`` offsets."""

    algorithm = "crop_resize"
    parameters = {"fraction": None, "top": 0, "left": 0}
    reversible = False  # content outside the window is gone

    @override
    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        fraction = self.params["fraction"]
        top, left = self.params["top"], self.params["left"]
        height, width = xs.shape[-2], xs.shape[-1]
        window_h, window_w = max(1, int(height * fraction)), max(1, int(width * fraction))
        window = xs[..., top : top + window_h, left : left + window_w]

        # nearest-neighbour back to the original size: deterministic, no dependency
        rows = np.arange(height) * window_h // height
        cols = np.arange(width) * window_w // width
        return window[..., rows[:, None], cols[None, :]]
