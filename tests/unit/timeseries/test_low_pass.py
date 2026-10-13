"""Semantics of the ``low_pass`` transformation.

The synthetic windows are a low and a high sinusoid, each on an exact FFT bin:
a cutoff between them must return the low part alone, a cutoff at Nyquist must
return the input, and the window space (shape, dtype) never changes. The class
comes from ``kcai_data_sampling_timeseries.api.transformations``.
"""

from kcai_data_sampling_timeseries.api.selection import TimeSeriesBatch
from kcai_data_sampling_timeseries.api.transformations.low_pass import LowPass
from kcai_data_sampling_timeseries.transformations.low_pass import low_pass
import numpy as np
from pydantic import ValidationError
import pytest

from tests.fixtures.series import HIGH_FREQUENCY, LOW_FREQUENCY, make_windows


def _out_of(batch: TimeSeriesBatch, cutoff: float) -> np.ndarray:
    """Transform one batch with ``LowPass`` and return the stacked outputs."""
    outputs = LowPass({"cutoff": cutoff}).transform(batch, batch.value_range)
    return np.stack([o.x for o in outputs])


def test_cutoff_between_the_components_keeps_the_low_one(
    synthetic_series_batch: TimeSeriesBatch, test_seed: int
) -> None:
    """A cutoff between 0.05 and 0.3 removes the high sinusoid exactly."""
    _, low_part = make_windows(test_seed)
    cutoff = (LOW_FREQUENCY + HIGH_FREQUENCY) / 2
    assert np.allclose(_out_of(synthetic_series_batch, cutoff), low_part)


def test_cutoff_below_both_components_keeps_the_mean(synthetic_series_batch: TimeSeriesBatch) -> None:
    """Below the low sinusoid only the constant offset (frequency 0) survives."""
    filtered = _out_of(synthetic_series_batch, LOW_FREQUENCY / 2)
    means = synthetic_series_batch.data.mean(axis=1, keepdims=True)
    assert np.allclose(filtered, np.broadcast_to(means, filtered.shape))


def test_nyquist_cutoff_is_identity(synthetic_series_batch: TimeSeriesBatch) -> None:
    """``cutoff=0.5`` keeps every frequency: the output equals the input."""
    assert np.allclose(_out_of(synthetic_series_batch, 0.5), synthetic_series_batch.data)


def test_shape_and_dtype_are_kept() -> None:
    """A float32 window stack comes back float32, same shape."""
    xs = np.random.default_rng(0).normal(size=(3, 64, 2)).astype(np.float32)
    out = low_pass(xs, 0.1)
    assert out.shape == xs.shape
    assert out.dtype == np.float32


def test_odd_window_length_is_supported() -> None:
    """The inverse FFT is told the window length, so odd widths round-trip."""
    xs = np.random.default_rng(0).normal(size=(2, 63, 1))
    assert np.allclose(low_pass(xs, 0.5), xs)


def test_integer_windows_are_refused() -> None:
    """Integer input would be truncated on the way back: it is refused."""
    with pytest.raises(ValueError, match="floating-point"):
        low_pass(np.zeros((1, 8, 1), dtype=np.int16), 0.1)


def test_non_window_shape_is_refused() -> None:
    """Anything but ``(B, W, C)`` is refused before the FFT."""
    with pytest.raises(ValueError, match=r"\(B, W, C\)"):
        low_pass(np.zeros((8, 1)), 0.1)


def test_declaration() -> None:
    """Deterministic, procedural, unary, not reversible, params complete."""
    assert LowPass({"cutoff": 0.1}).describe() == {
        "algorithm": "low_pass",
        "family": "procedural",
        "arity": "unary",
        "reversible": False,
        "params": {"cutoff": 0.1},
        "seed": None,
        "tool_model": None,
        "target_model": None,
    }


@pytest.mark.parametrize("cutoff", [0, -0.1, 0.6])
def test_cutoff_outside_zero_nyquist_is_refused(cutoff: float) -> None:
    """``cutoff`` must stay in ``(0, 0.5]``."""
    with pytest.raises(ValidationError, match="cutoff"):
        LowPass({"cutoff": cutoff})


def test_sweep_leaving_the_range_is_refused() -> None:
    """A sweep interval crossing Nyquist is refused at config load."""
    with pytest.raises(ValidationError, match="cutoff"):
        LowPass.Config.model_validate({"cutoff": {"range": [0.1, 0.7], "samples": 3}})


def test_low_pass_is_independent_of_the_batch_split(synthetic_series_batch: TimeSeriesBatch) -> None:
    """Window by window gives the same values as the whole batch."""
    whole = _out_of(synthetic_series_batch, 0.1)
    for i in range(len(synthetic_series_batch)):
        assert np.array_equal(_out_of(synthetic_series_batch.row(i), 0.1)[0], whole[i])
