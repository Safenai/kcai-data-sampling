"""Generative inpainting with LaMa.

Ships the ``inpaint`` transformation and its ``lama_inpaint`` tool model: the
transformation erases a rectangle of each image and a LaMa artifact fills the
hole with content that was not in the image. The transformation class lives
under ``api/transformations/``, the model under ``api/models/``, and the pure
mask math under ``transformations/``. ``torch`` is needed only here; the
checkpoint downloads into the weight cache (``$KCAI_WEIGHTS_DIR`` or the
repository's ``.cache/``) on first use.
"""

from ._version_ import __version__