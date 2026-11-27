"""``SweepConfig``: interval validation, exactly-one-of ``samples``/``step``/``values``, and value expansion.

``samples`` and ``step`` cover an ordered ``[min, max]`` interval; ``values``
lists the values and takes no interval. ``even`` expands by ``linspace``
(inclusive ends), ``step`` walks the arithmetic progression up to ``max``
included, ``random`` draws seeded uniform values — refused, loudly, without a
seed — and ``values`` gives them as listed.
"""

from kcai_data_sampling_core.models.sweep import SweepConfig, _coerce_to_field, expand_sweeps
from kcai_data_sampling_images.configs import CropResizeTransformationConfig
import pytest


def test_sweep_refuses_an_inverted_interval() -> None:
    """``min > max`` is refused at validation, naming the interval."""
    with pytest.raises(ValueError, match=r"sweep range \[1\.0, 0\.0\] must satisfy min <= max"):
        SweepConfig.model_validate({"range": [1, 0], "samples": 3})


def test_sweep_refuses_an_unknown_mode() -> None:
    """A typo is rejected instead of silently taking the random branch."""
    with pytest.raises(ValueError, match="mode"):
        SweepConfig.model_validate({"range": [0, 1], "samples": 3, "mode": "evne"})


def test_sweep_requires_exactly_one_of_samples_step_or_values() -> None:
    """None set, or two set, are refused with the same message."""
    message = r"exactly one of 'samples', 'step' or 'values'"
    with pytest.raises(ValueError, match=message):
        SweepConfig.model_validate({"range": [0, 1]})
    with pytest.raises(ValueError, match=message):
        SweepConfig.model_validate({"range": [0, 1], "samples": 3, "step": 0.5})
    with pytest.raises(ValueError, match=message):
        SweepConfig.model_validate({"values": [0.1, 0.2], "samples": 2})


def test_sweep_samples_and_step_need_a_range() -> None:
    """Without ``range``, ``samples`` and ``step`` have nothing to cover."""
    with pytest.raises(ValueError, match=r"'samples' or 'step' needs a 'range'"):
        SweepConfig.model_validate({"samples": 3})
    with pytest.raises(ValueError, match=r"'samples' or 'step' needs a 'range'"):
        SweepConfig.model_validate({"step": 0.1})


def test_sweep_step_walks_the_arithmetic_progression() -> None:
    """A step sweep yields ``min, min+step, ...``, dropping a partial tail."""
    sweep = SweepConfig.model_validate({"range": [0, 1], "step": 0.3})
    assert sweep.values() == [0.0, 0.3, 0.6, 0.9]


def test_sweep_step_reaches_the_upper_bound_despite_float_division() -> None:
    """``0.3 / 0.1`` is ``2.9999999999999996`` in floats: the walk still ends on 0.3."""
    assert SweepConfig.model_validate({"range": [0, 0.3], "step": 0.1}).values() == [0.0, 0.1, 0.2, 0.3]
    assert SweepConfig.model_validate({"range": [0, 0.7], "step": 0.1}).values()[-1] == 0.7
    assert SweepConfig.model_validate({"range": [1, 2], "step": 0.25}).values() == [1.0, 1.25, 1.5, 1.75, 2.0]


def test_sweep_values_are_used_as_listed() -> None:
    """``values`` gives irregular values, a log scale here, in the order written."""
    sweep = SweepConfig.model_validate({"values": [0.01, 0.001, 0.1]})
    assert sweep.values() == [0.01, 0.001, 0.1]
    assert sweep.bounds == (0.001, 0.1)


def test_sweep_values_refuse_a_range_an_empty_list_and_repeats() -> None:
    """``values`` stands alone, holds at least one value, and no value twice."""
    with pytest.raises(ValueError, match=r"'values' takes no 'range'"):
        SweepConfig.model_validate({"values": [0.1, 0.2], "range": [0, 1]})
    with pytest.raises(ValueError, match=r"at least 1 item"):
        SweepConfig.model_validate({"values": []})
    with pytest.raises(ValueError, match=r"sweep values \[0\.1, 0\.2, 0\.1\] repeat a value"):
        SweepConfig.model_validate({"values": [0.1, 0.2, 0.1]})


def test_sweep_values_expand_into_one_transformation_each() -> None:
    """In a transformation config, ``values`` expands like any sweep, with suffixed names."""
    config = CropResizeTransformationConfig(type="crop_resize", name="crop", fraction={"values": [0.9, 0.5]})
    variants = expand_sweeps(config)
    assert [v.fraction for v in variants] == [0.9, 0.5]
    assert [v.name for v in variants] == ["crop__0.9", "crop__0.5"]


def test_sweep_values_are_checked_against_the_parameter_bounds() -> None:
    """Every listed value must respect the parameter's bounds; the message shows the list."""
    with pytest.raises(ValueError, match=r"crop_resize\.fraction sweep \[0\.5, 1\.2\] must stay within \(0, 1\]"):
        CropResizeTransformationConfig(type="crop_resize", fraction={"values": [0.5, 1.2]})


def test_sweep_even_expands_linspace_inclusive() -> None:
    """``samples`` evenly covers the interval, ends included."""
    sweep = SweepConfig.model_validate({"range": [0.1, 0.5], "samples": 5})
    assert sweep.values() == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_sweep_keeps_small_values_by_default() -> None:
    """No rounding by default: slopes of a few thousandths stay distinct."""
    sweep = SweepConfig.model_validate({"range": [-0.02, -0.002], "samples": 6})
    assert sweep.values() == [-0.02, -0.0164, -0.0128, -0.0092, -0.0056, -0.002]
    assert SweepConfig.model_validate({"range": [0.0001, 0.0003], "samples": 3}).values() == [0.0001, 0.0002, 0.0003]


def test_sweep_drops_float_noise_by_default() -> None:
    """``linspace`` and the step walk lose their float noise (``0.15000000000000002``), not their precision."""
    assert SweepConfig.model_validate({"range": [0.05, 0.25], "samples": 3}).values() == [0.05, 0.15, 0.25]
    assert SweepConfig.model_validate({"range": [0, 1], "samples": 4}).values()[1] == pytest.approx(1 / 3, rel=1e-12)


def test_sweep_round_keeps_the_requested_decimals() -> None:
    """``round`` set: the values are rounded to that many decimals."""
    sweep = SweepConfig.model_validate({"range": [0, 1], "samples": 4, "round": 2})
    assert sweep.values() == [0.0, 0.33, 0.67, 1.0]


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
