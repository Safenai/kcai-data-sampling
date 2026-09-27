"""FGSM smoke: the opt-in adversarial wheel, no torch at import.

Runs inside the ``all+fgsm`` venv (all five kcai wheels): the ``-fgsm`` package
imports and exposes its transformation plugin and ``fgsm_step``, the
integer-only refusal fires, and a stub-target ``Fgsm`` round-trip runs on an
in-memory RGBA batch — all without ever importing torch or ultralytics. Like
the other opt-in packages, the target model is the user's: this venv proves
the wheel itself never pulls a model backend in.
"""

import sys

import numpy as np


def main() -> int:
    from kcai_data_sampling_fgsm.api.transformations.fgsm import Fgsm
    from kcai_data_sampling_fgsm.transformations.fgsm import fgsm_step

    # The opt-in surface never imports torch (or a YOLO backend) at import time.
    assert "torch" not in sys.modules
    assert "ultralytics" not in sys.modules

    # The integer-only refusal is loud and names the dtype.
    try:
        fgsm_step(np.zeros((2, 4, 4, 3), dtype=np.float64), np.ones((2, 4, 4, 3)), 1 / 255)
    except ValueError as error:
        assert "integer batches only" in str(error)
    else:
        raise AssertionError("fgsm_step accepted a float batch")

    # A uint8 sign-step round-trips shape/dtype and stays in range.
    frame = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    frame[..., 0] = 100
    grad = np.ones_like(frame, dtype=np.float64)
    perturbed = fgsm_step(frame, grad, 2 / 255)
    assert perturbed.shape == frame.shape and perturbed.dtype == np.uint8
    assert perturbed.min() >= 0 and perturbed.max() <= 255

    class StubTarget:
        name = "stub"

        def grad(self, xs):
            out = np.zeros(xs.shape, dtype=np.float64)
            out[..., 0] = 1.0
            return out

    fgsm = Fgsm({"target_model": StubTarget(), "epsilon": 2 / 255})
    rgba = np.zeros((4, 8, 8, 4), dtype=np.uint8)
    rgba[..., :3] = 200
    rgba[..., 3] = 255
    out = fgsm.apply(rgba, None)
    assert out.shape == rgba.shape and out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    assert out[..., 0].min() == 202  # every red pixel stepped by exactly +2

    print("smoke_fgsm ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())