"""Unary time-series transformation classes.

The generic core contract lives in ``kcai_data_sampling_core.api``; these are
the time-series subclasses, each wrapping a pure algorithm function from
``kcai_data_sampling_timeseries.transformations`` into the unary ``apply``
contract.
"""

from kcai_data_sampling_timeseries.api.transformations.add_sinusoid import AddSinusoid
from kcai_data_sampling_timeseries.api.transformations.colored_noise import ColoredNoise
from kcai_data_sampling_timeseries.api.transformations.drift import Drift
from kcai_data_sampling_timeseries.api.transformations.garch_noise import GarchNoise
from kcai_data_sampling_timeseries.api.transformations.impulse_noise import ImpulseNoise
from kcai_data_sampling_timeseries.api.transformations.jitter import Jitter
from kcai_data_sampling_timeseries.api.transformations.level_shift import LevelShift
from kcai_data_sampling_timeseries.api.transformations.low_pass import LowPass
from kcai_data_sampling_timeseries.api.transformations.ou_noise import OUNoise

__all__ = [
    "AddSinusoid",
    "ColoredNoise",
    "Drift",
    "GarchNoise",
    "ImpulseNoise",
    "Jitter",
    "LevelShift",
    "LowPass",
    "OUNoise",
]
