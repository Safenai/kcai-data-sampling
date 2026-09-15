"""Inpainting: generative.

A rectangle of the sample is erased and filled in by a tool model, which
produces content that was not in ``x``.
"""

import numpy as np
from typing_extensions import override

from kcai_data_sampling_core.api.unary import UnaryTransformation


class Inpaint(UnaryTransformation):
    """Params: ``top``, ``left``, ``height``, ``width`` of the region to erase, in pixels.
    Requires ``tool_model`` with ``inpaint(xs, masks)``."""

    algorithm = "inpaint"
    model_role = "tool"
    reversible = False  # what was in the region is gone; what replaces it is invented

    @override
    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        p = self.params
        masks = np.zeros((len(xs), *xs.shape[-2:]), dtype=bool)
        masks[:, p["top"] : p["top"] + p["height"], p["left"] : p["left"] + p["width"]] = True
        return self.tool_model.inpaint(xs, masks)
