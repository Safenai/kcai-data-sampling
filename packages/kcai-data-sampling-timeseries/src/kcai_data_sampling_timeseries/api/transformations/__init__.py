"""Unary time-series transformation classes.

The generic core contract lives in ``kcai_data_sampling_core.api``; these are
the time-series subclasses, each wrapping a pure algorithm function from
``kcai_data_sampling_timeseries.transformations`` into the unary ``apply``
contract.
"""

from kcai_data_sampling_timeseries.api.transformations.add_sinusoid import AddSinusoid
from kcai_data_sampling_timeseries.api.transformations.low_pass import LowPass

__all__ = ["AddSinusoid", "LowPass"]
