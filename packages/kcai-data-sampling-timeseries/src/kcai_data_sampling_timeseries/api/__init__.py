"""Time-series subclasses of the generic core contracts.

The generic core leaves the sample's space open (``sample_axes``,
``value_range``); this package holds the time-series specializations —
:class:`TimeSeriesBatch`, and the transformation classes. I/O stays out: reads
happen in the ``-job`` dataloaders and writes in the ``-job`` outputwriters;
everything flows as ``(B, W, C)`` float windows.
"""

from kcai_data_sampling_timeseries.api.selection import TimeSeriesBatch

__all__ = ["TimeSeriesBatch"]
