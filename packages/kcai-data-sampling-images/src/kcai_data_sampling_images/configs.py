"""Per-algorithm config schemas for the image transformations.

Each schema subclasses the core base ``TransformationConfig`` and pins
``type`` to the algorithm's literal, so the registry-resolved validator in
``JobConfig`` picks the right schema by `type`:
``fraction`` is required on ``crop_resize``, unknown fields are refused
everywhere.

A parameter that can sweep accepts ``SweepConfig`` in its schema; the ranges
the algorithm allows are enforced here, on both the plain value and the sweep
interval (``crop_resize.fraction`` must stay within ``(0, 1]``, ``top``/``left``
within ``[0, +oo)``).
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from kcai_data_sampling_core.models.config import TransformationConfig
from kcai_data_sampling_core.models.sweep import SweepConfig


class HorizontalFlipTransformationConfig(TransformationConfig):
    """Configuration of the ``horizontal_flip`` transformation.

    Holds the base keys only; the algorithm needs no parameters.
    """

    type: Literal["horizontal_flip"] = "horizontal_flip"


class CropResizeTransformationConfig(TransformationConfig):
    """Configuration of the ``crop_resize`` transformation.

    Attributes:
        fraction: Fraction ``(0, 1]`` of the image kept by the crop, or a
            ``SweepConfig`` expanding it (its interval must stay inside
            ``(0, 1]``).
        top: Crop window top offset in pixels (default 0), or a sweep (its
            interval must be ``>= 0``).
        left: Crop window left offset in pixels (default 0), or a sweep (its
            interval must be ``>= 0``).
    """

    type: Literal["crop_resize"] = "crop_resize"
    fraction: float | SweepConfig = Field(
        description="Fraction (0, 1] of the image kept; a SweepConfig expands it."
    )
    top: int | SweepConfig = Field(
        default=0,
        description="Crop window top offset in pixels; a SweepConfig expands it.",
    )
    left: int | SweepConfig = Field(
        default=0,
        description="Crop window left offset in pixels; a SweepConfig expands it.",
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce each parameter's algorithm range on the value or the sweep.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``fraction`` leaves ``(0, 1]``, or ``top``/``left``
                go below ``0``, whether given directly or as a sweep interval.
        """
        if isinstance(self.fraction, SweepConfig):
            lo, hi = self.fraction.bounds
            if lo <= 0 or hi > 1:
                raise ValueError(
                    f"crop_resize.fraction sweep {self.fraction.range} must stay within (0, 1]"
                )
        elif not 0 < self.fraction <= 1:
            raise ValueError(f"crop_resize.fraction must be in (0, 1], got {self.fraction}")
        for name in ("top", "left"):
            value = getattr(self, name)
            if isinstance(value, SweepConfig):
                lo, _ = value.bounds
                if lo < 0:
                    raise ValueError(
                        f"crop_resize.{name} sweep {value.range} must stay >= 0"
                    )
            elif value < 0:
                raise ValueError(f"crop_resize.{name} must be >= 0, got {value}")
        return self