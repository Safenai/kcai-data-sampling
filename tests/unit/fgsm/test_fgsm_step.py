"""The FGSM step's math, in normalized ``[0, 1]`` units.

``fgsm_step`` is the pure numpy map the ``Fgsm`` transformation wraps: it must
refuse non-integer batches (normalization by a dtype maximum has no meaning
otherwise), round-trip the dtype, move each pixel by exactly
``round(epsilon * iinfo.max)`` in the sign direction of the gradient, clip at
the integer extremes, depend on the sign only (never the magnitude), and work
for every integer dtype through ``np.iinfo``.
"""

import numpy as np
import pytest

pytest.importorskip("kcai_data_sampling_fgsm")

from kcai_data_sampling_fgsm.transformations.fgsm import fgsm_step


def test_integer_batches_only() -> None:
    """A float batch is refused: normalization by a dtype max needs integers."""
    xs = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="integer batches only"):
        fgsm_step(xs, np.zeros_like(xs), 2 / 255)


def test_uint8_round_trip_keeps_shape_and_dtype() -> None:
    """The step preserves shape and dtype exactly (uint8 in, uint8 out)."""
    xs = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = fgsm_step(xs, np.ones_like(xs, dtype=np.float64), 2 / 255)
    assert out.shape == xs.shape
    assert out.dtype == np.uint8


def test_delta_is_round_of_epsilon_times_the_max_on_positive_sign() -> None:
    """On the sign direction, the delta is exactly ``round(epsilon * 255)``.

    A mid-scale uint8 pixel plus a positive sign must move by
    ``round(epsilon * 255)`` — the 255 constant folded into the pixel units. Two
    budgets whose values land on exact integers (``2/255``, ``3/255``) pin the
    rounding rule unambiguously.
    """
    xs = (np.ones((1, 4), dtype=np.uint8) * 100).reshape(1, 2, 2, 1)
    grad = np.ones(xs.shape, dtype=np.float64)
    assert int(fgsm_step(xs, grad, 2 / 255)[0, 0, 0, 0]) == 102
    assert int(fgsm_step(xs, grad, 3 / 255)[0, 0, 0, 0]) == 103


def test_negative_sign_moves_the_other_way() -> None:
    """A negative sign moves the pixel down by the same exact delta."""
    xs = (np.ones((1, 4), dtype=np.uint8) * 100).reshape(1, 2, 2, 1)
    grad = -np.ones(xs.shape, dtype=np.float64)
    assert int(fgsm_step(xs, grad, 2 / 255)[0, 0, 0, 0]) == 98


def test_clip_at_zero_and_255_on_a_crossing_ramp() -> None:
    """Pixels pushed past the extremes clip back to ``0`` and ``255``.

    The 0..255 ramp with a sign that flips at the middle drives the low half
    toward ``0`` and the high half toward ``255``; clipped pixels land exactly
    on the extremes, the rest move by the exact delta.
    """
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 16, 16, 1)
    grad = (ramp.astype(np.float64) - 127.5)  # negative half, positive half
    out = fgsm_step(ramp, grad, 2 / 255)
    assert int(out[0, 0, 0, 0]) == 0  # 0 - 2 clipped at 0
    assert int(out[0, 0, 2, 0]) == 0  # 2 - 2 = 0, exactly
    assert int(out[0, 15, 15, 0]) == 255  # 255 + 2 clipped at 255
    assert int(out[0, 8, 0, 0]) == 130  # 128 + 2 mid-crossing


def test_sign_only_magnitude_is_irrelevant() -> None:
    """Two gradients with the same sign yield the identical step.

    The ``sign`` is the only thing taken from the gradient: a gentle and a
    steep gradient aligned in direction perturb the same pixels by the same
    (budget-limited) delta.
    """
    xs = np.arange(50, dtype=np.uint8).reshape(2, 5, 5, 1)
    gentle = np.where(xs > 20, 0.001, -0.001)
    steep = np.where(xs > 20, 40.0, -40.0)
    assert np.array_equal(fgsm_step(xs, gentle, 3 / 255), fgsm_step(xs, steep, 3 / 255))


def test_other_integer_dtypes_normalize_by_their_iinfo_max() -> None:
    """uint16 and uint32 normalize through ``np.iinfo``, not a hard-coded 255.

    The delta scales with the dtype's max: ``round(epsilon * 65535)`` moves a
    uint16 pixel, and a uint32 pixel normalizes by its own 2³²−1 maximum.
    """
    u16 = (np.ones((1, 1, 1, 1), dtype=np.uint16) * 30000)
    out16 = fgsm_step(u16, np.ones(u16.shape, dtype=np.float64), 2 / 65535)
    assert out16.dtype == np.uint16
    assert int(out16[0, 0, 0, 0]) == 30002

    u32 = (np.ones((1, 1, 1, 1), dtype=np.uint32) * 2_000_000_000)
    out32 = fgsm_step(u32, np.ones(u32.shape, dtype=np.float64), 2 / 4_294_967_295)
    assert out32.dtype == np.uint32
    assert int(out32[0, 0, 0, 0]) == 2_000_000_002