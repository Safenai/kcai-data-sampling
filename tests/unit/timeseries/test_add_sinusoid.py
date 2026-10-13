"""Semantics of the ``add_sinusoid`` transformation.

The added wave is ``amplitude * sin(2π * frequency * t + phase)`` with ``t``
counted from each window's first step, the same on every channel; the window
space (shape, dtype) never changes, and nothing is clipped since a series has
no value range. The class comes from
``kcai_data_sampling_timeseries.api.transformations``.
"""

from kcai_data_sampling_timeseries.api.selection import TimeSeriesBatch
from kcai_data_sampling_timeseries.api.transformations.add_sinusoid import AddSinusoid
from kcai_data_sampling_timeseries.transformations.add_sinusoid import add_sinusoid
import numpy as np
import pyarrow as pa
from pydantic import ValidationError
import pytest


def _out_of(batch: TimeSeriesBatch, **params: float) -> np.ndarray:
    """Transform one batch with ``AddSinusoid`` and return the stacked outputs."""
    outputs = AddSinusoid(params).transform(batch, batch.value_range)
    return np.stack([o.x for o in outputs])


def test_the_added_wave_is_exact(synthetic_series_batch: TimeSeriesBatch) -> None:
    """Output minus input is the requested sinusoid, on every window and channel."""
    out = _out_of(synthetic_series_batch, frequency=0.1, amplitude=2.0, phase=0.5)
    steps = np.arange(synthetic_series_batch.data.shape[1])
    wave = 2.0 * np.sin(2 * np.pi * 0.1 * steps + 0.5)
    assert np.allclose(out - synthetic_series_batch.data, wave[None, :, None])


def test_zero_amplitude_is_identity(synthetic_series_batch: TimeSeriesBatch) -> None:
    """``amplitude=0`` adds nothing."""
    out = _out_of(synthetic_series_batch, frequency=0.2, amplitude=0.0)
    assert np.array_equal(out, synthetic_series_batch.data)


def test_large_values_are_not_clipped() -> None:
    """With no value range, an output far from the input is kept as is."""
    xs = np.zeros((1, 4, 1))
    batch = TimeSeriesBatch(name="s", dataset="d", ids=["a@3"], columns=pa.table({"series_id": ["a"]}), data=xs)
    out = _out_of(batch, frequency=0.25, amplitude=1e6)
    assert np.isclose(out.max(), 1e6)


def test_shape_and_dtype_are_kept() -> None:
    """A float32 window stack comes back float32, same shape."""
    xs = np.zeros((2, 16, 3), dtype=np.float32)
    out = add_sinusoid(xs, frequency=0.1, amplitude=1.0)
    assert out.shape == xs.shape
    assert out.dtype == np.float32


def test_integer_windows_are_refused() -> None:
    """Integer input is refused rather than truncated."""
    with pytest.raises(ValueError, match="floating-point"):
        add_sinusoid(np.zeros((1, 8, 1), dtype=np.int32), frequency=0.1, amplitude=1.0)


def test_declaration_and_default_phase() -> None:
    """Deterministic, procedural, not reversible; ``phase`` defaults to 0."""
    assert AddSinusoid({"frequency": 0.1, "amplitude": 1.0}).describe() == {
        "algorithm": "add_sinusoid",
        "family": "procedural",
        "arity": "unary",
        "reversible": False,
        "params": {"frequency": 0.1, "amplitude": 1.0, "phase": 0.0},
        "seed": None,
        "tool_model": None,
        "target_model": None,
    }


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"frequency": 0, "amplitude": 1.0}, "frequency"),
        ({"frequency": 0.51, "amplitude": 1.0}, "frequency"),
        ({"frequency": 0.1, "amplitude": -1.0}, "amplitude"),
    ],
)
def test_parameters_outside_their_range_are_refused(params: dict[str, float], field: str) -> None:
    """``frequency`` stays in ``(0, 0.5]`` and ``amplitude`` is ``>= 0``."""
    with pytest.raises(ValidationError, match=field):
        AddSinusoid(params)


def test_missing_amplitude_is_refused() -> None:
    """``amplitude`` has no default: omitting it is a config error."""
    with pytest.raises(ValidationError, match="amplitude"):
        AddSinusoid({"frequency": 0.1})
