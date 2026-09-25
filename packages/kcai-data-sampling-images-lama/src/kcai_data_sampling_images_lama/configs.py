"""Per-algorithm config schemas for the inpaint transformation.

The schema subclasses the core base ``TransformationConfig`` and pins ``type``
to the algorithm's literal, so the registry-resolved validator in ``JobConfig``
picks it by `type`. A parameter that can sweep accepts ``SweepConfig``; the
ranges the algorithm allows are enforced here, on both the plain value and the
sweep interval (``top``/``left`` within ``[0, +oo)``, ``height``/``width``
within ``(0, +oo)``). ``tool_model`` names a model from the ``models:`` section
of the job config; the CLI resolves the name to an instance.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from kcai_data_sampling_core.models.config import TransformationConfig
from kcai_data_sampling_core.models.sweep import SweepConfig


class InpaintTransformationConfig(TransformationConfig):
    """Configuration of the ``inpaint`` transformation.

    Attributes:
        tool_model: Name of a tool model from the job's ``models:`` section;
            the CLI resolves it to an instance. Optional here because the
            transformation base consumes it before re-validating the remaining
            parameters against this schema; a model-role transformation without
            one is refused loudly at construction (the slot check).
        top: Region top offset in pixels (default 0), or a sweep (its interval
            must be ``>= 0``).
        left: Region left offset in pixels (default 0), or a sweep (its
            interval must be ``>= 0``).
        height: Region height in pixels; required (a degenerate empty region is
            refused), or a sweep (its interval must be ``> 0``).
        width: Region width in pixels; required (a degenerate empty region is
            refused), or a sweep (its interval must be ``> 0``).
    """

    type: Literal["inpaint"] = "inpaint"
    tool_model: str | None = Field(
        default=None,
        exclude=True,
        description="Name of a tool model from the job's models: section; excluded"
        " from the resolved parameters (the base consumes it before validation).",
    )
    top: int | SweepConfig = Field(
        default=0,
        description="Region top offset in pixels; a SweepConfig expands it.",
    )
    left: int | SweepConfig = Field(
        default=0,
        description="Region left offset in pixels; a SweepConfig expands it.",
    )
    height: int | SweepConfig = Field(
        description="Region height in pixels; a SweepConfig expands it."
    )
    width: int | SweepConfig = Field(
        description="Region width in pixels; a SweepConfig expands it."
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce each parameter's algorithm range on the value or the sweep.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``top``/``left`` go below ``0`` or
                ``height``/``width`` do not stay positive, whether given
                directly or as a sweep interval.
        """
        for name in ("top", "left"):
            self._bound(name, minimum=0, exclusive=False)
        for name in ("height", "width"):
            self._bound(name, minimum=0, exclusive=True)
        return self

    def _bound(self, name: str, minimum: int, exclusive: bool) -> None:
        """Refuse a value or sweep interval outside the parameter's range.

        Args:
            name: The parameter name, for the error message.
            minimum: The smallest allowed value (inclusive unless ``exclusive``).
            exclusive: Whether the bound is exclusive (a strictly positive
                parameter).
        """
        value = getattr(self, name)
        if isinstance(value, SweepConfig):
            lo, _ = value.bounds
            if lo < minimum or (exclusive and lo == minimum):
                comparator = ">" if exclusive else ">="
                raise ValueError(
                    f"inpaint.{name} sweep {value.range} must stay {comparator} {minimum}"
                )
            return
        if value < minimum or (exclusive and value == minimum):
            comparator = ">" if exclusive else ">="
            raise ValueError(f"inpaint.{name} must be {comparator} {minimum}, got {value}")