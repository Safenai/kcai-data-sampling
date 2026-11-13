"""Semantics of the ``drift`` transformation.

The added ramp is ``slope * t`` with ``t`` counted from each window's first
step, the same on every channel; the window space (shape, dtype) never changes,
and nothing is clipped since a series has no value range. The class comes from
``kcai_data_sampling_timeseries.api.transformations``.
"""

from kcai_data_sampling_timeseries.api.selection import TimeSeriesBatch
from kcai_data_sampling_timeseries.api.transformations.drift import Drift
from kcai_data_sampling_timeseries.transformations.drift import drift
import numpy as np
from pydantic import ValidationError
import pytest


def _out_of(batch: TimeSeriesBatch, **params: float) -> np.ndarray:
    """Transform one batch with ``Drift`` and return the stacked outputs."""
    outputs = Drift(params).transform(batch, batch.value_range)
    return np.stack([o.x for o in outputs])


@pytest.mark.parametrize("slope", [0.05, -0.2])
def test_the_added_ramp_is_exact(synthetic_series_batch: TimeSeriesBatch, slope: float) -> None:
    """Output minus input is ``slope * t``, on every window and channel, either sign."""
    out = _out_of(synthetic_series_batch, slope=slope)
    steps = np.arange(synthetic_series_batch.data.shape[1])
    assert np.allclose(out - synthetic_series_batch.data, slope * steps[None, :, None])


def test_the_ramp_starts_at_zero_on_the_first_step(synthetic_series_batch: TimeSeriesBatch) -> None:
    """The first step of each window is left unchanged; the last moves by ``slope * (W - 1)``."""
    out = _out_of(synthetic_series_batch, slope=0.5)
    data = synthetic_series_batch.data
    assert np.array_equal(out[:, 0], data[:, 0])
    assert np.allclose(out[:, -1] - data[:, -1], 0.5 * (data.shape[1] - 1))


def test_zero_slope_is_identity(synthetic_series_batch: TimeSeriesBatch) -> None:
    """``slope=0`` adds nothing."""
    out = _out_of(synthetic_series_batch, slope=0.0)
    assert np.array_equal(out, synthetic_series_batch.data)


def test_shape_and_dtype_are_kept() -> None:
    """A float32 window stack comes back float32, same shape."""
    xs = np.zeros((2, 16, 3), dtype=np.float32)
    out = drift(xs, slope=0.1)
    assert out.shape == xs.shape
    assert out.dtype == np.float32


def test_integer_windows_are_refused() -> None:
    """Integer input is refused rather than truncated."""
    with pytest.raises(ValueError, match="floating-point"):
        drift(np.zeros((1, 8, 1), dtype=np.int32), slope=0.1)


def test_declaration() -> None:
    """Deterministic, procedural, not reversible; no seed."""
    assert Drift({"slope": 0.1}).describe() == {
        "algorithm": "drift",
        "family": "procedural",
        "arity": "unary",
        "reversible": False,
        "params": {"slope": 0.1},
        "seed": None,
        "tool_model": None,
        "target_model": None,
    }


def test_missing_slope_is_refused() -> None:
    """``slope`` has no default: omitting it is a config error."""
    with pytest.raises(ValidationError, match="slope"):
        Drift({})
