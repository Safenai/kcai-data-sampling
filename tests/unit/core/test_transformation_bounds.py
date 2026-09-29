"""Transformation bound checks: unknown types and per-parameter ranges.

The shared bound machinery lives on the base ``TransformationConfig``; each
algorithm schema calls ``_check_parameter_bounds`` for its parameters. The
messages are exercised through ``crop_resize`` (a ``(0, 1]`` fraction and
non-negative window offsets), and the two-sided/upper-bound wording is pinned
directly so every message branch is covered.
"""

from kcai_data_sampling_core.models.config import TransformationConfig
from kcai_data_sampling_core.models.sweep import SweepConfig
from kcai_data_sampling_images.configs import CropResizeTransformationConfig
import pytest


def _crop(fraction, **extra):
    """Build a ``crop_resize`` config with extra parameters."""
    return CropResizeTransformationConfig(type="crop_resize", fraction=fraction, **extra)


def test_unknown_transformation_type_is_refused() -> None:
    """A ``type`` string outside the loaded registry fails at validation."""
    with pytest.raises(ValueError, match=r"unknown transformation type 'nope'"):
        TransformationConfig(type="nope")


def test_crop_resize_fraction_must_stay_within_the_open_lower_end() -> None:
    """``fraction`` of 0 (the exclusive end) is refused with the two-sided message."""
    with pytest.raises(ValueError, match=r"crop_resize\.fraction must be in \(0, 1\], got 0"):
        _crop(0)


def test_crop_resize_fraction_must_stay_below_the_inclusive_upper_end() -> None:
    """``fraction`` above 1 is refused with the same two-sided message."""
    with pytest.raises(ValueError, match=r"crop_resize\.fraction must be in \(0, 1\], got 1\.5"):
        _crop(1.5)


def test_crop_resize_window_offsets_cannot_go_negative() -> None:
    """A negative ``top`` violates the single lower bound with the right wording."""
    with pytest.raises(ValueError, match=r"crop_resize\.top must be >= 0, got -1"):
        _crop(0.5, top=-1)


def test_crop_resize_fraction_sweep_lower_end_is_enforced() -> None:
    """A swept fraction crossing the exclusive lower end is refused."""
    with pytest.raises(ValueError, match=r"crop_resize\.fraction sweep \[0\.0, 0\.5\] must stay within \(0, 1\]"):
        _crop(SweepConfig(range=[0, 0.5], samples=3))


def test_crop_resize_fraction_sweep_upper_end_is_enforced() -> None:
    """A swept fraction crossing the inclusive upper end is refused."""
    with pytest.raises(ValueError, match=r"crop_resize\.fraction sweep \[0\.2, 1\.2\] must stay within \(0, 1\]"):
        _crop(SweepConfig(range=[0.2, 1.2], samples=3))


def test_crop_resize_window_offset_sweep_lower_end_is_enforced() -> None:
    """A swept ``top`` going negative is refused with the single-bound wording."""
    with pytest.raises(ValueError, match=r"crop_resize\.top sweep \[-1\.0, 5\.0\] must stay >= 0"):
        _crop(0.5, top=SweepConfig(range=[-1, 5], samples=3))


def test_two_sided_message_with_an_exclusive_upper_end() -> None:
    """The ``(min, max)`` wording flips to hard parens for exclusive max."""
    config = _crop(0.5)
    with pytest.raises(ValueError, match=r"crop_resize\.fraction must be in \[0\.2, 0\.4\), got 0\.5"):
        config._check_parameter_bounds("fraction", minimum=0.2, maximum=0.4, exclusive_max=True)


def test_single_upper_bound_message() -> None:
    """A bare ``maximum`` yields ``must be <=``."""
    config = _crop(0.5)
    with pytest.raises(ValueError, match=r"crop_resize\.fraction must be <= 0\.2, got 0\.5"):
        config._check_parameter_bounds("fraction", maximum=0.2)


def test_sweep_single_upper_bound_message() -> None:
    """A sweep hitting a bare ``maximum`` yields ``must stay <=``."""
    config = _crop(SweepConfig(range=[0.2, 0.5], samples=3))
    with pytest.raises(ValueError, match=r"crop_resize\.fraction sweep \[0\.2, 0\.5\] must stay <= 0\.3"):
        config._check_parameter_bounds("fraction", maximum=0.3)
