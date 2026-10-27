"""The ``jitter`` transformation class.

Wraps the additive-noise math
(:func:`kcai_data_sampling_timeseries.transformations.jitter.jitter`) into the
unary transformation contract.
"""

from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_timeseries.configs import JitterTransformationConfig
from kcai_data_sampling_timeseries.transformations.jitter import jitter


class Jitter(UnaryTransformation):
    """Add Gaussian noise of standard deviation ``sigma`` to each window.

    Stochastic: the noise is drawn, so a seed is required; each window draws
    from its own generator, seeded by the seed, its parent id and the
    parameters, whatever the batch it lands in. Not reversible: the noise
    cannot be told apart from the signal.
    """

    algorithm = "jitter"

    #: The registered config schema this algorithm validates against.
    Config = JitterTransformationConfig

    #: The noise is drawn: a seed is required and recorded.
    stochastic = True

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, w, c) float
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Add the noise to the batch.

        Args:
            xs: Batch of windows shaped ``(B, W, C)``, floating point.
            rngs: One generator per window, provided by the base.

        Returns:
            The batch with the noise added, same shape and dtype.
        """
        assert rngs is not None
        return jitter(xs, sigma=self.params["sigma"], rngs=rngs)
