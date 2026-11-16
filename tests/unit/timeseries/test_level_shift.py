"""Semantics of the ``level_shift`` transformation.

The steps before ``floor(start * W)`` are unchanged and every step from it on
moves by ``offset``, the same on every channel; the window space (shape,
dtype) never changes, and nothing is clipped since a series has no value
range. The class comes from ``kcai_data_sampling_timeseries.api.transformations``.
"""

from kcai_data_sampling_timeseries.api.selection import TimeSeriesBatch
from kcai_data_sampling_timeseries.api.transformations.level_shift import LevelShift
from kcai_data_sampling_timeseries.transformations.level_shift import level_shift, shift_step
import numpy as np
from pydantic import ValidationError
import pytest


def _out_of(batch: TimeSeriesBatch, **params: float) -> np.ndarray:
    """Transform one batch with ``LevelShift`` and return the stacked outputs."""
    outputs = LevelShift(params).transform(batch, batch.value_range)
    return np.stack([o.x for o in outputs])


@pytest.mark.parametrize(("offset", "start"), [(2.0, 0.5), (-0.7, 0.25), (1.5, 0.9)])
def test_the_added_step_is_exact(synthetic_series_batch: TimeSeriesBatch, offset: float, start: float) -> None:
    """Output minus input is 0 before the break and ``offset`` from it on, on every window and channel."""
    out = _out_of(synthetic_series_batch, offset=offset, start=start)
    w = synthetic_series_batch.data.shape[1]
    expected = np.where(np.arange(w) >= shift_step(w, start), offset, 0.0)
    assert np.allclose(out - synthetic_series_batch.data, expected[None, :, None])


@pytest.mark.parametrize(
    ("length", "start", "step"), [(10, 0.0, 0), (10, 0.5, 5), (10, 0.55, 5), (7, 0.5, 3), (10, 1.0, 10)]
)
def test_the_break_position(length: int, start: float, step: int) -> None:
    """The break is at ``floor(start * W)``: 0 shifts the whole window, 1 shifts nothing."""
    assert shift_step(length, start) == step


def test_start_one_and_zero_offset_are_identity(synthetic_series_batch: TimeSeriesBatch) -> None:
    """``start=1`` reaches no step and ``offset=0`` adds nothing."""
    assert np.array_equal(_out_of(synthetic_series_batch, offset=3.0, start=1.0), synthetic_series_batch.data)
    assert np.array_equal(_out_of(synthetic_series_batch, offset=0.0), synthetic_series_batch.data)


def test_shape_and_dtype_are_kept() -> None:
    """A float32 window stack comes back float32, same shape."""
    xs = np.zeros((2, 16, 3), dtype=np.float32)
    out = level_shift(xs, offset=0.1)
    assert out.shape == xs.shape
    assert out.dtype == np.float32


def test_integer_windows_are_refused() -> None:
    """Integer input is refused rather than truncated."""
    with pytest.raises(ValueError, match="floating-point"):
        level_shift(np.zeros((1, 8, 1), dtype=np.int32), offset=0.1)


def test_declaration_and_default_start() -> None:
    """Deterministic, procedural, not reversible; ``start`` defaults to 0.5."""
    assert LevelShift({"offset": -0.2}).describe() == {
        "algorithm": "level_shift",
        "family": "procedural",
        "arity": "unary",
        "reversible": False,
        "params": {"offset": -0.2, "start": 0.5},
        "seed": None,
        "tool_model": None,
        "target_model": None,
    }


@pytest.mark.parametrize("start", [-0.1, 1.1])
def test_start_outside_the_window_is_refused(start: float) -> None:
    """``start`` stays in ``[0, 1]``."""
    with pytest.raises(ValidationError, match="start"):
        LevelShift({"offset": 1.0, "start": start})


def test_missing_offset_is_refused() -> None:
    """``offset`` has no default: omitting it is a config error."""
    with pytest.raises(ValidationError, match="offset"):
        LevelShift({})
