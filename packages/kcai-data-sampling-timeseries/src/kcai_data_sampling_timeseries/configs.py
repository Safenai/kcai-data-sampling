"""Per-algorithm config schemas for the time-series transformations.

Each schema subclasses the core base ``TransformationConfig`` and pins
``type`` to the algorithm's literal, so the registry-resolved validator in
``JobConfig`` picks the right schema by ``type``; unknown fields are refused
everywhere.

Frequencies are in cycles per step and must stay within ``(0, 0.5]``, the
range a sampled signal can represent; a parameter that can sweep accepts
``SweepConfig``, and the ranges are enforced on both the plain value and the
sweep interval.
"""

from typing import Literal, Self

from kcai_data_sampling_core.models.config import TransformationConfig
from kcai_data_sampling_core.models.sweep import SweepConfig
from pydantic import Field, model_validator

#: The highest frequency a sampled signal represents, in cycles per step.
NYQUIST = 0.5


class LowPassTransformationConfig(TransformationConfig):
    """Configuration of the ``low_pass`` transformation.

    Attributes:
        cutoff: Highest frequency kept, in cycles per step, in ``(0, 0.5]``,
            or a ``SweepConfig`` expanding it.
    """

    type: Literal["low_pass"] = "low_pass"
    cutoff: float | SweepConfig = Field(
        description="Highest frequency kept, cycles per step in (0, 0.5]; a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the cutoff range on the value or the sweep.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``cutoff`` leaves ``(0, 0.5]``.
        """
        self._check_parameter_bounds("cutoff", minimum=0, maximum=NYQUIST, exclusive_min=True)
        return self


class AddSinusoidTransformationConfig(TransformationConfig):
    """Configuration of the ``add_sinusoid`` transformation.

    Attributes:
        frequency: Frequency of the sinusoid, in cycles per step, in
            ``(0, 0.5]``, or a sweep.
        amplitude: Peak amplitude in the units of the signal, ``>= 0``, or a
            sweep.
        phase: Phase at the window's first step, in radians (default 0), or a
            sweep.
    """

    type: Literal["add_sinusoid"] = "add_sinusoid"
    frequency: float | SweepConfig = Field(
        description="Frequency, cycles per step in (0, 0.5]; a SweepConfig expands it.",
    )
    amplitude: float | SweepConfig = Field(
        description="Peak amplitude in signal units, >= 0; a SweepConfig expands it.",
    )
    phase: float | SweepConfig = Field(
        default=0.0,
        description="Phase at the window's first step, radians; a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce each parameter's range on the value or the sweep.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``frequency`` leaves ``(0, 0.5]`` or ``amplitude``
                goes below ``0``.
        """
        self._check_parameter_bounds("frequency", minimum=0, maximum=NYQUIST, exclusive_min=True)
        self._check_parameter_bounds("amplitude", minimum=0)
        return self


class JitterTransformationConfig(TransformationConfig):
    """Configuration of the ``jitter`` transformation.

    Attributes:
        sigma: Standard deviation of the added noise, in the units of the
            signal, ``>= 0``, or a sweep.
    """

    type: Literal["jitter"] = "jitter"
    sigma: float | SweepConfig = Field(
        description="Noise standard deviation in signal units, >= 0; a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the sigma range on the value or the sweep.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``sigma`` goes below ``0``.
        """
        self._check_parameter_bounds("sigma", minimum=0)
        return self
