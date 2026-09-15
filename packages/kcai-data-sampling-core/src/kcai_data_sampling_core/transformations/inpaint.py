"""Inpainting: generative.

A rectangle of the sample is erased and filled in by a tool model, which
produces content that was not in ``x``.
"""

import numpy as np
from typing_extensions import override

from kcai_data_sampling_core.api.roles import check_output
from kcai_data_sampling_core.api.unary import UnaryTransformation


class Inpaint(UnaryTransformation):
    """The region to erase, in pixels: ``top``, ``left``, ``height``, ``width``."""

    algorithm = "inpaint"
    model_role = "tool"
    parameters = {"top": None, "left": None, "height": None, "width": None}
    requires = ("inpaint",)
    reversible = False  # what was in the region is gone; what replaces it is invented

    @override
    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        p = self.params
        masks = np.zeros((len(xs), *xs.shape[-2:]), dtype=bool)
        masks[:, p["top"] : p["top"] + p["height"], p["left"] : p["left"] + p["width"]] = True
        return check_output(self.algorithm, self.tool_model, "inpaint", xs, self.tool_model.inpaint(xs, masks))
