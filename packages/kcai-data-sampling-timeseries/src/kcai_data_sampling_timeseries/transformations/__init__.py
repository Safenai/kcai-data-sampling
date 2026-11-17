"""Unary time-series transformation math functions.

All time-series algorithms share the window convention ``(B, W, C)``, floating
point, with axis 1 as time. Frequencies are in cycles per step, in
``(0, 0.5]`` (0.5 is the Nyquist frequency). The module-level functions here
are the pure algorithm math; the transformation classes that wrap them into
the unary contract live in ``kcai_data_sampling_timeseries.api.transformations``.
"""

from kcai_data_sampling_timeseries.transformations.add_sinusoid import add_sinusoid
from kcai_data_sampling_timeseries.transformations.colored_noise import colored_noise
from kcai_data_sampling_timeseries.transformations.drift import drift
from kcai_data_sampling_timeseries.transformations.garch_noise import garch_noise
from kcai_data_sampling_timeseries.transformations.impulse_noise import impulse_noise
from kcai_data_sampling_timeseries.transformations.jitter import jitter
from kcai_data_sampling_timeseries.transformations.level_shift import level_shift
from kcai_data_sampling_timeseries.transformations.low_pass import low_pass
from kcai_data_sampling_timeseries.transformations.ou_noise import ou_noise

__all__ = [
    "add_sinusoid",
    "colored_noise",
    "drift",
    "garch_noise",
    "impulse_noise",
    "jitter",
    "level_shift",
    "low_pass",
    "ou_noise",
]
