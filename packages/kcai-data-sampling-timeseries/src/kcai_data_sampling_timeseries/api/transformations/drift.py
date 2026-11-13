"""The ``drift`` transformation class.

Wraps the linear-drift math
(:func:`kcai_data_sampling_timeseries.transformations.drift.drift`) into the
unary transformation contract.
"""

from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_timeseries.configs import DriftTransformationConfig
from kcai_data_sampling_timeseries.transformations.drift import drift


class Drift(UnaryTransformation):
    """Add a linear drift of ``slope`` per step to every channel of each window.

    Deterministic: the slope is a parameter, nothing is drawn. Not reversible:
    the added ramp cannot be told apart from a trend already present in the
    signal.
    """

    algorithm = "drift"

    #: The registered config schema this algorithm validates against.
    Config = DriftTransformationConfig

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, w, c) float
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Add the drift to the batch.

        Args:
            xs: Batch of windows shaped ``(B, W, C)``, floating point.
            rngs: Unused for this deterministic operation.

        Returns:
            The batch with the drift added, same shape and dtype.
        """
        del rngs
        return drift(xs, slope=self.params["slope"])
