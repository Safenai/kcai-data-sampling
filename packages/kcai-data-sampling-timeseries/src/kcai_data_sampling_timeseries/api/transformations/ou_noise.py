"""The ``ou_noise`` transformation class.

Wraps the Ornstein-Uhlenbeck noise math
(:func:`kcai_data_sampling_timeseries.transformations.ou_noise.ou_noise`) into the
unary transformation contract.
"""

from kcai_data_sampling_core.api.unary import UnaryTransformation
import numpy as np
from typing_extensions import override

from kcai_data_sampling_timeseries.configs import OUNoiseTransformationConfig
from kcai_data_sampling_timeseries.transformations.ou_noise import ou_noise


class OUNoise(UnaryTransformation):
    """Apply Ornstein-Uhlenbeck noise: mean-reverting at rate ``theta``, stationary deviation ``sigma``.

    Stochastic: the noise is drawn, so a seed is required; each window draws
    from its own generator, seeded by the seed, its parent id and the
    parameters, whatever the batch it lands in. Not reversible: the noise
    cannot be told apart from the signal.
    """

    algorithm = "ou_noise"

    #: The registered config schema this algorithm validates against.
    Config = OUNoiseTransformationConfig

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
        return ou_noise(xs, sigma=p["sigma"], theta=p["theta"], rngs=rngs, mode=p["mode"])
