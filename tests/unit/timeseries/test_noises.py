"""Semantics of the time-series noises: ``colored_noise``, ``ou_noise``, ``garch_noise``, ``impulse_noise``.

Each noise is checked on the property that defines it (spectrum slope,
mean reversion, volatility clustering, spike rate), on its two modes
(``x + n`` and ``x * (1 + n)``), on seeding (the base gives each window its own
generator) and on its config bounds.
"""

from kcai_data_sampling_timeseries.api.selection import TimeSeriesBatch
from kcai_data_sampling_timeseries.api.transformations import ColoredNoise, GarchNoise, ImpulseNoise, OUNoise
from kcai_data_sampling_timeseries.transformations import colored_noise, garch_noise, impulse_noise, ou_noise
import numpy as np
from pydantic import ValidationError
import pytest

NOISES = {
    "colored_noise": (colored_noise, ColoredNoise, {"sigma": 0.5, "exponent": 1.0}),
    "ou_noise": (ou_noise, OUNoise, {"sigma": 0.5, "theta": 0.2}),
    "garch_noise": (garch_noise, GarchNoise, {"sigma": 0.5, "alpha": 0.1, "beta": 0.85}),
    "impulse_noise": (impulse_noise, ImpulseNoise, {"rate": 0.05, "amplitude": 2.0}),
}


def _rngs(n: int, seed: int = 0) -> list[np.random.Generator]:
    return [np.random.default_rng(seed + i) for i in range(n)]


def _noise(name: str, shape: tuple[int, int, int], **params: float) -> np.ndarray:
    """The noise alone: the transformation applied additively to zeros."""
    fn, _, defaults = NOISES[name]
    return fn(np.zeros(shape), **{**defaults, **params}, rngs=_rngs(shape[0]))


@pytest.mark.parametrize("name", NOISES)
def test_shape_and_dtype_are_kept(name: str) -> None:
    """A float32 window stack comes back float32, same shape."""
    fn, _, params = NOISES[name]
    out = fn(np.zeros((2, 16, 3), dtype=np.float32), **params, rngs=_rngs(2))
    assert out.shape == (2, 16, 3)
    assert out.dtype == np.float32


@pytest.mark.parametrize("name", NOISES)
def test_multiplicative_is_relative_to_the_signal(name: str) -> None:
    """Same draws: multiplicative gives ``x * (1 + n)`` where additive gives ``x + n``."""
    fn, _, params = NOISES[name]
    x = np.full((2, 64, 2), 10.0)
    n = fn(np.zeros_like(x), **params, rngs=_rngs(2), mode="additive")
    assert np.allclose(fn(x, **params, rngs=_rngs(2), mode="additive"), x + n)
    assert np.allclose(fn(x, **params, rngs=_rngs(2), mode="multiplicative"), x * (1 + n))


@pytest.mark.parametrize("name", NOISES)
def test_unknown_mode_and_generator_count_are_refused(name: str) -> None:
    """A mode other than additive / multiplicative, or one generator short, is an error."""
    fn, cls, params = NOISES[name]
    with pytest.raises(ValueError, match="mode"):
        fn(np.zeros((1, 8, 1)), **params, rngs=_rngs(1), mode="replace")
    with pytest.raises(ValueError, match="one generator per window"):
        fn(np.zeros((2, 8, 1)), **params, rngs=_rngs(1))
    with pytest.raises(ValidationError, match="mode"):
        cls({**params, "seed": 0, "mode": "replace"})


@pytest.mark.parametrize("name", NOISES)
def test_seeded_and_independent_of_the_batch_split(name: str, synthetic_series_batch: TimeSeriesBatch) -> None:
    """Same seed, same draw; another seed, another draw; window by window equals the whole batch."""
    _, cls, params = NOISES[name]

    def run(batch: TimeSeriesBatch, seed: int) -> np.ndarray:
        return np.stack([o.x for o in cls({**params, "seed": seed}).transform(batch, batch.value_range)])

    whole = run(synthetic_series_batch, 0)
    assert np.array_equal(whole, run(synthetic_series_batch, 0))
    assert not np.array_equal(whole, run(synthetic_series_batch, 1))
    for i in range(len(synthetic_series_batch)):
        assert np.array_equal(run(synthetic_series_batch.row(i), 0)[0], whole[i])


@pytest.mark.parametrize("name", NOISES)
def test_declaration(name: str) -> None:
    """Stochastic, procedural, not reversible; the seed and the default mode are recorded."""
    _, cls, params = NOISES[name]
    described = cls({**params, "seed": 0}).describe()
    assert cls.stochastic
    assert described["seed"] == 0
    assert described["params"]["mode"] == "additive"
    assert (described["family"], described["arity"], described["reversible"]) == ("procedural", "unary", False)


@pytest.mark.parametrize("name", NOISES)
def test_a_seed_is_required(name: str) -> None:
    """A stochastic algorithm without a seed fails at construction."""
    _, cls, params = NOISES[name]
    with pytest.raises(ValueError, match="requires a seed"):
        cls(params)


# --- colored_noise


def test_colored_noise_has_exactly_sigma_and_no_mean() -> None:
    """Each window and channel is rescaled to standard deviation ``sigma``, with zero mean."""
    n = _noise("colored_noise", (3, 256, 2), sigma=0.7, exponent=1.0)
    assert np.allclose(n.std(axis=1), 0.7)
    assert np.allclose(n.mean(axis=1), 0.0, atol=1e-9)


@pytest.mark.parametrize("exponent", [0.0, 1.0, 2.0])
def test_colored_noise_spectrum_falls_as_one_over_f_to_the_exponent(exponent: float) -> None:
    """The log-log slope of the average power spectrum is ``-exponent``."""
    n = _noise("colored_noise", (64, 1024, 1), sigma=1.0, exponent=exponent)[:, :, 0]
    power = (np.abs(np.fft.rfft(n, axis=1)) ** 2).mean(axis=0)[1:]
    freqs = np.fft.rfftfreq(1024)[1:]
    slope = np.polyfit(np.log(freqs), np.log(power), 1)[0]
    assert abs(slope + exponent) < 0.1


def test_colored_noise_exponent_outside_zero_two_is_refused() -> None:
    """``exponent`` stays in ``[0, 2]``."""
    with pytest.raises(ValidationError, match="exponent"):
        ColoredNoise({"sigma": 1.0, "exponent": 2.5, "seed": 0})


# --- ou_noise


def test_ou_noise_reverts_with_the_expected_correlation_and_spread() -> None:
    """Stationary deviation ``sigma``; correlation between successive steps ``exp(-theta)``."""
    n = _noise("ou_noise", (400, 200, 1), sigma=0.5, theta=0.2)[:, :, 0]
    assert abs(n.std() - 0.5) < 0.02
    lag1 = np.corrcoef(n[:, :-1].ravel(), n[:, 1:].ravel())[0, 1]
    assert abs(lag1 - np.exp(-0.2)) < 0.02


def test_ou_noise_theta_must_be_positive() -> None:
    """``theta`` is a rate, strictly positive."""
    with pytest.raises(ValidationError, match="theta"):
        OUNoise({"sigma": 1.0, "theta": 0.0, "seed": 0})


# --- garch_noise


def test_garch_noise_has_sigma_and_clustered_volatility() -> None:
    """Unconditional deviation ``sigma``; uncorrelated values, but correlated squares."""
    n = _noise("garch_noise", (300, 400, 1), sigma=0.5, alpha=0.15, beta=0.8)[:, :, 0]
    assert abs(n.std() - 0.5) < 0.03
    assert abs(np.corrcoef(n[:, :-1].ravel(), n[:, 1:].ravel())[0, 1]) < 0.03
    assert np.corrcoef((n[:, :-1] ** 2).ravel(), (n[:, 1:] ** 2).ravel())[0, 1] > 0.1


@pytest.mark.parametrize(
    "params",
    [
        {"sigma": 1.0, "alpha": 0.5, "beta": 0.5},
        {"sigma": 1.0, "alpha": {"range": [0.1, 0.3], "samples": 3}, "beta": 0.75},
    ],
)
def test_garch_noise_alpha_plus_beta_must_stay_below_one(params: dict) -> None:
    """Refused for a value, and for a sweep whose largest values would reach 1."""
    with pytest.raises(ValidationError, match="alpha \\+ beta"):
        GarchNoise({**params, "seed": 0})
    with pytest.raises(ValueError, match="alpha \\+ beta"):
        garch_noise(np.zeros((1, 8, 1)), sigma=1.0, alpha=0.5, beta=0.5, rngs=_rngs(1))


# --- impulse_noise


def test_impulse_noise_hits_at_the_rate_with_signed_spikes() -> None:
    """About ``rate`` of the steps are hit, each by exactly ``±amplitude``, both signs."""
    n = _noise("impulse_noise", (50, 400, 2), rate=0.05, amplitude=2.0)
    hits = n != 0
    assert abs(hits.mean() - 0.05) < 0.005
    assert set(np.unique(n[hits])) == {-2.0, 2.0}


def test_impulse_noise_rate_zero_is_identity() -> None:
    """``rate=0`` adds nothing."""
    x = np.random.default_rng(0).normal(size=(2, 32, 1))
    assert np.array_equal(impulse_noise(x, rate=0.0, amplitude=5.0, rngs=_rngs(2)), x)


def test_impulse_noise_rate_outside_zero_one_is_refused() -> None:
    """``rate`` is a probability."""
    with pytest.raises(ValidationError, match="rate"):
        ImpulseNoise({"rate": 1.5, "amplitude": 1.0, "seed": 0})
