"""The ``low_pass`` transformation class.

Wraps the FFT low-pass math
(:func:`kcai_data_sampling_timeseries.transformations.low_pass.low_pass`) into
the unary transformation contract.
"""

from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_timeseries.configs import LowPassTransformationConfig
from kcai_data_sampling_timeseries.transformations.low_pass import low_pass


class LowPass(UnaryTransformation):
    """Zero the frequencies above ``cutoff`` in each window (ideal FFT filter).

    Deterministic, and not reversible: the removed frequencies are gone.
    """

    algorithm = "low_pass"

    #: The registered config schema this algorithm validates against.
    Config = LowPassTransformationConfig

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, w, c) float
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Filter the batch.

        Args:
            xs: Batch of windows shaped ``(B, W, C)``, floating point.
            rngs: Unused for this deterministic operation.

        Returns:
            The filtered batch, same shape and dtype.
        """
        del rngs
        return low_pass(xs, cutoff=self.params["cutoff"])
