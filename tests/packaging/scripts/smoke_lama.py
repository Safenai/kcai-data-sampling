"""Lama smoke: the opt-in wheel surface, no torch at import.

Runs inside the ``all+lama`` venv (all five kcai wheels + CPU torch): the
``-lama`` package imports and exposes its model plugin, and a stub-tool
``Inpaint`` round-trip runs on an in-memory image batch — all without ever
importing torch. torch is installed in this venv exactly to prove ``-lama``
never pulls it in itself: the checkpoint load in ``LamaTool()`` is the only
place torch is needed, and it stays deferred.
"""

import contextlib
import io
import sys

import numpy as np


def main() -> int:
    from kcai_data_sampling_lama.api.models.lama import URL, LamaTool
    from kcai_data_sampling_lama.api.transformations.inpaint import Inpaint

    # The opt-in surface never imports torch at import time.
    assert "torch" not in sys.modules

    # `version` resolves this member: it is installed here, so it must not read
    # as absent the way it does in the lama-free `all` scenario.
    from kcai_data_sampling.dependency import display_version

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        display_version()
    assert "lama: None" not in out.getvalue()

    # The plugin's declared class surface on the wheel.
    assert LamaTool.channels == 3
    assert URL.endswith("big-lama.pt")

    class StubTool:
        name = "stub"

        def inpaint(self, xs, masks):
            out = xs.copy()
            out[..., :3][masks] = 0
            return out

    region = {"top": 2, "left": 2, "height": 4, "width": 4}
    inpaint = Inpaint(config={"seed": None, "tool_model": StubTool(), **region})

    frames = np.zeros((4, 8, 8, 4), dtype=np.uint8)
    frames[..., :3] = 200
    frames[..., 3] = 255
    out = inpaint.apply(frames, None)
    assert out.shape == frames.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255

    top, left, height, width = region.values()
    window = (slice(top, top + height), slice(left, left + width))
    assert not out[:, window[0], window[1], :3].any()
    assert np.array_equal(out[:, window[0], window[1], 3], frames[:, window[0], window[1], 3])
    outside = np.ones((8, 8), dtype=bool)
    outside[window] = False
    assert np.array_equal(out[:, outside, :3], frames[:, outside, :3])
    print("smoke_lama ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
