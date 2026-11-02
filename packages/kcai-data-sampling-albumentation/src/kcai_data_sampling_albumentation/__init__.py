"""Albumentations-backed procedural image transformations.

Ships one registered transformation type per upstream albumentations
transform, all sharing a single engine class (``engine.AlbumentationsUnary``)
that maps the workspace's unary contract onto a bare upstream
``BasicTransform``. The transformation classes and their pydantic config
schemas are generated from the declarative allowlist in ``specs.py`` (see
``_generate.py``); the generation is committed, and a freshness test keeps it
honest. The upstream import is pinned and frozen — quarantined here so the
rest of the workspace stays free of OpenCV and of the pin.
"""

import os

# albumentations' import-time update check hits pypi.org unless this is set;
# set it before any submodule can import albumentations (engine.py repeats the
# guard defensively, since it may also be imported first in odd test orders).
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from kcai_data_sampling_albumentation._version_ import __version__ as __version__

__all__ = ["__version__"]
