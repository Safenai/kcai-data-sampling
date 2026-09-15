"""Horizontal flip, procedural."""

import numpy as np
from typing_extensions import override

from kcai_data_sampling_core.api.unary import UnaryTransformation


class HorizontalFlip(UnaryTransformation):
    """Mirror the sample along its width axis."""

    algorithm = "horizontal_flip"
    reversible = True

    @override
    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        return np.ascontiguousarray(xs[..., ::-1])
