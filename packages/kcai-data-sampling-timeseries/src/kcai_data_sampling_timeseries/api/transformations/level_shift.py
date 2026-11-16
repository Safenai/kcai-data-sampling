"""The ``level_shift`` transformation class.

Wraps the level-shift math
(:func:`kcai_data_sampling_timeseries.transformations.level_shift.level_shift`)
into the unary transformation contract.
"""

from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_timeseries.configs import LevelShiftTransformationConfig
from kcai_data_sampling_timeseries.transformations.level_shift import level_shift


class LevelShift(UnaryTransformation):
    """Shift every channel of each window by ``offset`` from ``start`` on.

    Deterministic: the offset and the position are parameters, nothing is
    drawn. Not reversible: the break cannot be told apart from a real change
    of level in the signal.
    """

    algorithm = "level_shift"

    #: The registered config schema this algorithm validates against.
    Config = LevelShiftTransformationConfig

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, w, c) float
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Add the shift to the batch.

        Args:
            xs: Batch of windows shaped ``(B, W, C)``, floating point.
            rngs: Unused for this deterministic operation.

        Returns:
            The batch with the shift added, same shape and dtype.
        """
        del rngs
        return level_shift(xs, offset=self.params["offset"], start=self.params["start"])
