"""The ``add_sinusoid`` transformation class.

Wraps the added-sinusoid math
(:func:`kcai_data_sampling_timeseries.transformations.add_sinusoid.add_sinusoid`)
into the unary transformation contract.
"""

from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_timeseries.configs import AddSinusoidTransformationConfig
from kcai_data_sampling_timeseries.transformations.add_sinusoid import add_sinusoid


class AddSinusoid(UnaryTransformation):
    """Add a sinusoid of a given frequency to every channel of each window.

    Deterministic: the frequency, amplitude and phase are parameters, nothing
    is drawn. Not reversible: the added component cannot be told apart from
    the same frequency already present in the signal.
    """

    algorithm = "add_sinusoid"

    #: The registered config schema this algorithm validates against.
    Config = AddSinusoidTransformationConfig

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, w, c) float
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Add the sinusoid to the batch.

        Args:
            xs: Batch of windows shaped ``(B, W, C)``, floating point.
            rngs: Unused for this deterministic operation.

        Returns:
            The batch with the sinusoid added, same shape and dtype.
        """
        del rngs
        return add_sinusoid(
            xs,
            frequency=self.params["frequency"],
            amplitude=self.params["amplitude"],
            phase=self.params["phase"],
        )
