"""Semantics of the ``level_shift`` transformation.

Every step of every channel moves by ``offset``: the whole window, no position
parameter (a break at a given step will come from a ``support`` common to all
transformations). The window space (shape, dtype) never changes, and nothing is
clipped since a series has no value range. The class comes from
``kcai_data_sampling_timeseries.api.transformations``.
"""

from kcai_data_sampling_timeseries.api.selection import TimeSeriesBatch
from kcai_data_sampling_timeseries.api.transformations.level_shift import LevelShift
from kcai_data_sampling_timeseries.transformations.level_shift import level_shift
import numpy as np
from pydantic import ValidationError
import pytest


def _out_of(batch: TimeSeriesBatch, **params: float) -> np.ndarray:
    """Transform one batch with ``LevelShift`` and return the stacked outputs."""
    outputs = LevelShift(params).transform(batch, batch.value_range)
    return np.stack([o.x for o in outputs])


@pytest.mark.parametrize("offset", [2.0, -0.7, 1.5])
def test_the_whole_window_moves_by_offset(synthetic_series_batch: TimeSeriesBatch, offset: float) -> None:
    """Output minus input is ``offset`` on every step, window and channel."""
    out = _out_of(synthetic_series_batch, offset=offset)
    assert np.allclose(out - synthetic_series_batch.data, offset)


def test_zero_offset_is_identity(synthetic_series_batch: TimeSeriesBatch) -> None:
    """``offset=0`` adds nothing."""
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


def test_declaration() -> None:
    """Deterministic, procedural, not reversible; ``offset`` is the only parameter."""
    assert LevelShift({"offset": -0.2}).describe() == {
        "algorithm": "level_shift",
        "family": "procedural",
        "arity": "unary",
        "reversible": False,
        "params": {"offset": -0.2},
        "seed": None,
        "tool_model": None,
        "target_model": None,
    }


def test_a_position_is_refused() -> None:
    """There is no ``start``: the shift applies to the whole window."""
    with pytest.raises(ValidationError, match="start"):
        LevelShift({"offset": 1.0, "start": 0.5})


def test_missing_offset_is_refused() -> None:
    """``offset`` has no default: omitting it is a config error."""
    with pytest.raises(ValidationError, match="offset"):
        LevelShift({})
