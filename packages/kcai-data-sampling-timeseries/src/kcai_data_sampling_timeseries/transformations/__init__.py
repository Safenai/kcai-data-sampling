"""Unary time-series transformation math functions.

All time-series algorithms share the window convention ``(B, W, C)``, floating
point, with axis 1 as time. Frequencies are in cycles per step, in
``(0, 0.5]`` (0.5 is the Nyquist frequency). The module-level functions here
are the pure algorithm math; the transformation classes that wrap them into
the unary contract live in ``kcai_data_sampling_timeseries.api.transformations``.
"""

from kcai_data_sampling_timeseries.transformations.add_sinusoid import add_sinusoid
from kcai_data_sampling_timeseries.transformations.jitter import jitter
from kcai_data_sampling_timeseries.transformations.low_pass import low_pass

__all__ = ["add_sinusoid", "jitter", "low_pass"]
