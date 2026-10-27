"""Semantics of the ``jitter`` transformation.

The added noise is ``N(0, sigma²)``, drawn independently per step and channel
from one generator per window. Checked: its statistics, that ``sigma = 0`` is
the identity, that the seed is required, recorded and decides the draw, that
each window's draw never depends on the batch it lands in, and that the window
space (shape, dtype) never changes. The class comes from
``kcai_data_sampling_timeseries.api.transformations``.
"""

from kcai_data_sampling_timeseries.api.selection import TimeSeriesBatch
from kcai_data_sampling_timeseries.api.transformations.jitter import Jitter
from kcai_data_sampling_timeseries.transformations.jitter import jitter
import numpy as np
from pydantic import ValidationError
import pytest


def _out_of(batch: TimeSeriesBatch, **params: float) -> np.ndarray:
    """Transform one batch with ``Jitter`` and return the stacked outputs."""
    outputs = Jitter(params).transform(batch, batch.value_range)
    return np.stack([o.x for o in outputs])


def test_noise_has_zero_mean_and_the_requested_deviation() -> None:
    """Over many steps the noise averages to 0 with standard deviation ``sigma``."""
    xs = np.zeros((4, 5000, 2))
    noise = jitter(xs, sigma=0.5, rngs=[np.random.default_rng(i) for i in range(4)])
    assert abs(noise.mean()) < 0.01
    assert abs(noise.std() - 0.5) < 0.01


def test_each_window_and_channel_draws_its_own_noise() -> None:
    """Two windows, or two channels, never receive the same noise."""
    noise = jitter(np.zeros((2, 64, 2)), sigma=1.0, rngs=[np.random.default_rng(i) for i in range(2)])
    assert not np.array_equal(noise[0], noise[1])
    assert not np.array_equal(noise[0, :, 0], noise[0, :, 1])


def test_zero_sigma_is_identity(synthetic_series_batch: TimeSeriesBatch) -> None:
    """``sigma=0`` adds nothing."""
    out = _out_of(synthetic_series_batch, sigma=0.0, seed=1)
    assert np.array_equal(out, synthetic_series_batch.data)


def test_the_seed_decides_the_draw(synthetic_series_batch: TimeSeriesBatch) -> None:
    """Same seed, same outputs; another seed, other outputs."""
    first = _out_of(synthetic_series_batch, sigma=0.1, seed=3)
    assert np.array_equal(first, _out_of(synthetic_series_batch, sigma=0.1, seed=3))
    assert not np.array_equal(first, _out_of(synthetic_series_batch, sigma=0.1, seed=4))


def test_jitter_is_independent_of_the_batch_split(synthetic_series_batch: TimeSeriesBatch) -> None:
    """Window by window gives the same draw as the whole batch."""
    whole = _out_of(synthetic_series_batch, sigma=0.1, seed=3)
    for i in range(len(synthetic_series_batch)):
        assert np.array_equal(_out_of(synthetic_series_batch.row(i), sigma=0.1, seed=3)[0], whole[i])


def test_shape_and_dtype_are_kept() -> None:
    """A float32 window stack comes back float32, same shape."""
    xs = np.zeros((2, 16, 3), dtype=np.float32)
    out = jitter(xs, sigma=1.0, rngs=[np.random.default_rng(0), np.random.default_rng(1)])
    assert out.shape == xs.shape
    assert out.dtype == np.float32


def test_integer_windows_are_refused() -> None:
    """Integer input is refused rather than truncated."""
    with pytest.raises(ValueError, match="floating-point"):
        jitter(np.zeros((1, 8, 1), dtype=np.int32), sigma=1.0, rngs=[np.random.default_rng(0)])


def test_one_generator_per_window_is_required() -> None:
    """A generator count that does not match the batch is refused."""
    with pytest.raises(ValueError, match="one generator per window"):
        jitter(np.zeros((2, 8, 1)), sigma=1.0, rngs=[np.random.default_rng(0)])


def test_declaration_records_the_seed() -> None:
    """Stochastic, procedural, not reversible; the seed reaches the row."""
    transformation = Jitter({"sigma": 0.1, "seed": 7})
    assert transformation.stochastic
    assert transformation.describe() == {
        "algorithm": "jitter",
        "family": "procedural",
        "arity": "unary",
        "reversible": False,
        "params": {"sigma": 0.1},
        "seed": 7,
        "tool_model": None,
        "target_model": None,
    }


def test_missing_seed_is_refused() -> None:
    """A stochastic algorithm without a seed fails at construction."""
    with pytest.raises(ValueError, match="requires a seed"):
        Jitter({"sigma": 0.1})


@pytest.mark.parametrize("params", [{"sigma": -0.1, "seed": 1}, {"seed": 1}])
def test_bad_or_missing_sigma_is_refused(params: dict[str, float]) -> None:
    """``sigma`` is required and ``>= 0``."""
    with pytest.raises(ValidationError, match="sigma"):
        Jitter(params)
