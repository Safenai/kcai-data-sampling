"""The ``impulse_noise`` transformation class.

Wraps the impulse-noise math
(:func:`kcai_data_sampling_timeseries.transformations.impulse_noise.impulse_noise`) into the
unary transformation contract.
"""

from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_timeseries.configs import ImpulseNoiseTransformationConfig
from kcai_data_sampling_timeseries.transformations.impulse_noise import impulse_noise


class ImpulseNoise(UnaryTransformation):
    """Apply isolated spikes of size ``amplitude``, each step hit with probability ``rate``.

    Stochastic: the noise is drawn, so a seed is required; each window draws
    from its own generator, seeded by the seed, its parent id and the
    parameters, whatever the batch it lands in. Not reversible: the noise
    cannot be told apart from the signal.
    """

    algorithm = "impulse_noise"

    #: The registered config schema this algorithm validates against.
    Config = ImpulseNoiseTransformationConfig

    #: The noise is drawn: a seed is required and recorded.
    stochastic = True

    @override
    def apply(
        self,
        xs: np.ndarray,  # (b, w, c) float
        rngs: list[np.random.Generator] | None = None,
    ) -> np.ndarray:
        """Apply the noise to the batch.

        Args:
            xs: Batch of windows shaped ``(B, W, C)``, floating point.
            rngs: One generator per window, provided by the base.

        Returns:
            The noisy batch, same shape and dtype.
        """
        assert rngs is not None
        p = self.params
        return impulse_noise(xs, rate=p["rate"], amplitude=p["amplitude"], rngs=rngs, mode=p["mode"])
