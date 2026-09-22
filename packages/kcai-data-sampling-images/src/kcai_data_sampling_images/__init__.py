"""No-model image transformations and image data IO.

Ships the phase-1 image algorithm set (``horizontal_flip``, ``crop_resize``),
the ``image_dir`` dataloader, and the ``images`` payload writer. Everything
flows as ``(B, H, W, 4)`` uint8 RGBA arrays.
"""

from ._version_ import __version__