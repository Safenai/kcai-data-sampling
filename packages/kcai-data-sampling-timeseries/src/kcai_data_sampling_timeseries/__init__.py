"""No-model time-series transformations.

Ships the time-series algorithm set (``low_pass``, ``add_sinusoid``): the math
functions under ``transformations/`` and the transformation classes under
``api/transformations/``, plus the ``TimeSeriesBatch`` specialization under
``api/``. I/O is deliberately absent: reads happen in the ``-job`` dataloaders
and writes in the ``-job`` outputwriters; everything flows as ``(B, W, C)``
float windows.
"""

from kcai_data_sampling_timeseries._version_ import __version__ as __version__
