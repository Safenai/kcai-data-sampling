"""Albumentation smoke: the opt-in wheel surface, offline at import.

Runs inside the ``all+albumentation`` venv (all five kcai wheels + the pinned
albumentations): the ``-albumentation`` package imports with the PyPI update
check suppressed, exposes both transformation plugins, and a
``RandomBrightnessContrast`` round-trip runs on an in-memory RGBA batch —
alpha untouched, uint8 in range, one output per row. Nothing here needs a
network or a dataset: the pinned upstream is a wheel-level dependency only.
"""

import contextlib
import io
import os
import sys

import numpy as np


def main() -> int:
    from kcai_data_sampling.dependency import display_version
    from kcai_data_sampling_albumentation.api.transformations import RandomBrightnessContrast

    # `version` resolves this member: it is installed here, so it must not read
    # as absent the way it does in the albumentation-free `all` scenario.
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        display_version()
    assert "albumentation: None" not in out.getvalue()

    # The import-time PyPI update check is suppressed by the package guard.
    assert os.environ.get("NO_ALBUMENTATIONS_UPDATE") == "1"

    # The wheel's own entry points survived publication.
    from kcai_data_sampling_core.utils.registry import get_transformations_registry

    registry = get_transformations_registry()
    for name in ("gaussian_blur", "random_brightness_contrast"):
        assert registry[name].__module__.startswith("kcai_data_sampling_albumentation")

    # An RGBA round-trip through the unary contract: the image-only transform
    # moves RGB only, the alpha plane comes back byte-identical.
    frames = np.random.default_rng(0).integers(0, 256, size=(2, 8, 8, 4), dtype=np.uint8)
    from kcai_data_sampling_core.api.selection import Batch
    import pyarrow as pa

    batch = Batch(name="smoke", dataset="smoke", ids=["a", "b"], columns=pa.table({"tag": ["a", "b"]}), data=frames)
    transform = RandomBrightnessContrast({"seed": 42, "brightness_limit": [0.5, 0.5], "contrast_limit": [0.0, 0.0]})
    result = np.stack([output.x for output in transform.transform(batch, value_range=(0, 255))])
    assert result.shape == frames.shape
    assert result.dtype == np.uint8
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255
    assert np.array_equal(result[..., 3], frames[..., 3])
    assert not np.array_equal(result[..., :3], frames[..., :3])

    print("smoke_albumentation ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
