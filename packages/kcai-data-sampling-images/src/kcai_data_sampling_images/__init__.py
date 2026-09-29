"""No-model image transformations.

Ships the image algorithm set (``horizontal_flip``, ``crop_resize``): the
math functions under ``transformations/`` and the transformation classes under
``api/transformations/``, plus the ``ImageBatch`` specialization under
``api/``. I/O is deliberately absent: reads happen in the ``-job`` dataloaders
and writes in the ``-job`` outputwriters; everything flows as ``(B, H, W, 4)``
uint8 RGBA arrays.
"""

from kcai_data_sampling_images._version_ import __version__ as __version__
