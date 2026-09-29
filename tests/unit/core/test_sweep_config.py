"""``SweepConfig``: interval validation, exactly-one-of ``samples``/``step``, and value expansion.

An ordered ``[min, max]`` interval is required; ``samples`` and ``step`` are
mutually exclusive. ``even`` expands by ``linspace`` (inclusive ends), ``step``
walks the arithmetic progression, and ``random`` draws seeded uniform values —
refused, loudly, without a seed.
"""

from kcai_data_sampling_core.models.sweep import _coerce_to_field, SweepConfig
import pytest


def test_sweep_refuses_an_inverted_interval() -> None:
    """``min > max`` is refused at validation, naming the interval."""
    with pytest.raises(ValueError, match=r"sweep range \[1\.0, 0\.0\] must satisfy min <= max"):
        SweepConfig.model_validate({"range": [1, 0], "samples": 3})


def test_sweep_requires_exactly_one_of_samples_or_step() -> None:
    """Both unset and both set are refused with the same message."""
    with pytest.raises(ValueError, match=r"exactly one of 'samples' or 'step'"):
        SweepConfig.model_validate({"range": [0, 1]})
    with pytest.raises(ValueError, match=r"exactly one of 'samples' or 'step'"):
        SweepConfig.model_validate({"range": [0, 1], "samples": 3, "step": 0.5})


def test_sweep_step_walks_the_arithmetic_progression() -> None:
    """A step sweep yields ``min, min+step, ...``, dropping a partial tail."""
    sweep = SweepConfig.model_validate({"range": [0, 1], "step": 0.3})
    assert sweep.values() == [0.0, 0.3, 0.6, 0.9]


def test_sweep_even_expands_linspace_inclusive() -> None:
    """``samples`` evenly covers the interval, ends included."""
    sweep = SweepConfig.model_validate({"range": [0.1, 0.5], "samples": 5})
    assert sweep.values() == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_sweep_random_refuses_to_run_without_a_seed() -> None:
    """``mode: random`` without a seed would be non-deterministic; refused."""
    sweep = SweepConfig.model_validate({"range": [0, 1], "samples": 3, "mode": "random"})
    with pytest.raises(ValueError, match=r"random needs the compute seed"):
        sweep.values()


def test_sweep_random_with_a_seed_draws_the_requested_count_in_range() -> None:
    """A seeded random sweep is reproducible and bounded by the interval."""
    first = SweepConfig.model_validate({"range": [2, 5], "samples": 4, "mode": "random"})
    drawn = first.values(seed=7)
    assert len(drawn) == 4
    assert all(2.0 <= value <= 5.0 for value in drawn)
    assert drawn == SweepConfig.model_validate({"range": [2, 5], "samples": 4, "mode": "random"}).values(seed=7)


def test_sweep_bounds_reports_the_interval() -> None:
    """``bounds`` exposes the validated ``(min, max)`` pair."""
    sweep = SweepConfig.model_validate({"range": [2, 5], "samples": 2})
    assert sweep.bounds == (2.0, 5.0)


def test_coerce_to_field_casts_integers_for_int_accepting_fields() -> None:
    """A union accepting ``int`` receives an int; a float-only field does not."""
    assert _coerce_to_field(2.9, int | float) == 2
    assert _coerce_to_field(2.9, float) == 2.9
