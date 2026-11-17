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


class DriftTransformationConfig(TransformationConfig):
    """Configuration of the ``drift`` transformation.

    Attributes:
        slope: Change added per step, in the units of the signal, any sign,
            or a sweep.
    """

    type: Literal["drift"] = "drift"
    slope: float | SweepConfig = Field(
        description="Change added per step in signal units, any sign; a SweepConfig expands it.",
    )


class LevelShiftTransformationConfig(TransformationConfig):
    """Configuration of the ``level_shift`` transformation.

    Attributes:
        offset: Change of level, in the units of the signal, any sign, or a
            sweep.
        start: Position of the break as a fraction of the window, in
            ``[0, 1]`` (default 0.5, halfway), or a sweep.
    """

    type: Literal["level_shift"] = "level_shift"
    offset: float | SweepConfig = Field(
        description="Change of level in signal units, any sign; a SweepConfig expands it.",
    )
    start: float | SweepConfig = Field(
        default=0.5,
        description="Position of the break as a fraction of the window, in [0, 1]; a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the start range on the value or the sweep.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``start`` leaves ``[0, 1]``.
        """
        self._check_parameter_bounds("start", minimum=0, maximum=1)
        return self


def _upper(value: "float | SweepConfig") -> float:
    """The largest value a parameter can take: the value itself, or the top of its sweep."""
    return value.bounds[1] if isinstance(value, SweepConfig) else value


class ColoredNoiseTransformationConfig(TransformationConfig):
    """Configuration of the ``colored_noise`` transformation.

    Attributes:
        sigma: Standard deviation of the noise, ``>= 0`` (signal units when
            additive, a relative change when multiplicative), or a sweep.
        exponent: Spectral exponent in ``[0, 2]``: 0 white, 1 pink (default),
            2 brown; or a sweep.
        mode: ``additive`` (``x + n``, default) or ``multiplicative``
            (``x * (1 + n)``).
    """

    type: Literal["colored_noise"] = "colored_noise"
    sigma: float | SweepConfig = Field(description="Noise standard deviation, >= 0; a SweepConfig expands it.")
    exponent: float | SweepConfig = Field(
        default=1.0, description="Spectral exponent in [0, 2]: 0 white, 1 pink, 2 brown; a SweepConfig expands it."
    )
    mode: Literal["additive", "multiplicative"] = Field(
        default="additive", description="additive: x + n; multiplicative: x * (1 + n)."
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the sigma and exponent ranges on the value or the sweep.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``sigma`` goes below 0 or ``exponent`` leaves ``[0, 2]``.
        """
        self._check_parameter_bounds("sigma", minimum=0)
        self._check_parameter_bounds("exponent", minimum=0, maximum=2)
        return self


class OUNoiseTransformationConfig(TransformationConfig):
    """Configuration of the ``ou_noise`` (Ornstein-Uhlenbeck) transformation.

    Attributes:
        sigma: Stationary standard deviation, ``>= 0``, or a sweep.
        theta: Mean-reversion rate per step, ``> 0`` (``1 / theta`` is the
            correlation time in steps), or a sweep.
        mode: ``additive`` (default) or ``multiplicative``.
    """

    type: Literal["ou_noise"] = "ou_noise"
    sigma: float | SweepConfig = Field(description="Stationary standard deviation, >= 0; a SweepConfig expands it.")
    theta: float | SweepConfig = Field(description="Mean-reversion rate per step, > 0; a SweepConfig expands it.")
    mode: Literal["additive", "multiplicative"] = Field(
        default="additive", description="additive: x + n; multiplicative: x * (1 + n)."
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the sigma and theta ranges on the value or the sweep.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``sigma`` goes below 0 or ``theta`` is not above 0.
        """
        self._check_parameter_bounds("sigma", minimum=0)
        self._check_parameter_bounds("theta", minimum=0, exclusive_min=True)
        return self


class GarchNoiseTransformationConfig(TransformationConfig):
    """Configuration of the ``garch_noise`` (GARCH(1,1)) transformation.

    Attributes:
        sigma: Unconditional standard deviation, ``>= 0``, or a sweep.
        alpha: Weight of the last shock, ``>= 0`` (default 0.1), or a sweep.
        beta: Persistence of the variance, ``>= 0`` (default 0.85), or a sweep;
            ``alpha + beta`` must stay below 1, sweeps included.
        mode: ``additive`` (default) or ``multiplicative``.
    """

    type: Literal["garch_noise"] = "garch_noise"
    sigma: float | SweepConfig = Field(description="Unconditional standard deviation, >= 0; a SweepConfig expands it.")
    alpha: float | SweepConfig = Field(
        default=0.1, description="Weight of the last shock, >= 0; a SweepConfig expands it."
    )
    beta: float | SweepConfig = Field(
        default=0.85, description="Persistence of the variance, >= 0; a SweepConfig expands it."
    )
    mode: Literal["additive", "multiplicative"] = Field(
        default="additive", description="additive: x + n; multiplicative: x * (1 + n)."
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the ranges, and ``alpha + beta < 1`` for the largest values a sweep can reach.

        Returns:
            The validated config.

        Raises:
            ValueError: If a parameter leaves its range or ``alpha + beta`` can reach 1.
        """
        self._check_parameter_bounds("sigma", minimum=0)
        self._check_parameter_bounds("alpha", minimum=0)
        self._check_parameter_bounds("beta", minimum=0)
        if _upper(self.alpha) + _upper(self.beta) >= 1:
            raise ValueError(
                f"garch_noise: alpha + beta must stay below 1, got up to {_upper(self.alpha) + _upper(self.beta)}"
            )
        return self


class ImpulseNoiseTransformationConfig(TransformationConfig):
    """Configuration of the ``impulse_noise`` (spikes, salt and pepper) transformation.

    Attributes:
        rate: Probability of a spike at each step, in ``[0, 1]``, or a sweep.
        amplitude: Size of a spike, ``>= 0`` (signal units when additive, a
            relative change when multiplicative), or a sweep.
        mode: ``additive`` (default) or ``multiplicative``.
    """

    type: Literal["impulse_noise"] = "impulse_noise"
    rate: float | SweepConfig = Field(
        description="Probability of a spike at each step, in [0, 1]; a SweepConfig expands it."
    )
    amplitude: float | SweepConfig = Field(description="Size of a spike, >= 0; a SweepConfig expands it.")
    mode: Literal["additive", "multiplicative"] = Field(
        default="additive", description="additive: x + n; multiplicative: x * (1 + n)."
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce the rate and amplitude ranges on the value or the sweep.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``rate`` leaves ``[0, 1]`` or ``amplitude`` goes below 0.
        """
        self._check_parameter_bounds("rate", minimum=0, maximum=1)
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
