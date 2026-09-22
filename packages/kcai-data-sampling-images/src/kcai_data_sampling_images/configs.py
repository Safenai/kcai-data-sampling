"""Per-algorithm config schemas for the image transformations.

Each schema subclasses the core base ``TransformationConfig`` and pins
``type`` to the algorithm's literal, so the registry-resolved validator in
``JobConfig`` picks the right schema by `type`:
``fraction`` is required on ``crop_resize``, unknown fields are refused
everywhere.
"""

from typing import Literal

from pydantic import Field

from kcai_data_sampling_core.models.config import TransformationConfig


class HorizontalFlipTransformationConfig(TransformationConfig):
    """Configuration of the ``horizontal_flip`` transformation.

    Holds the base keys only; the algorithm needs no parameters.
    """

    type: Literal["horizontal_flip"] = "horizontal_flip"


class CropResizeTransformationConfig(TransformationConfig):
    """Configuration of the ``crop_resize`` transformation.

    Attributes:
        fraction: Fraction ``(0, 1]`` of the image kept by the crop.
        top: Crop window top offset in pixels (default 0).
        left: Crop window left offset in pixels (default 0).
    """

    type: Literal["crop_resize"] = "crop_resize"
    fraction: float = Field(gt=0, le=1, description="Fraction (0, 1] of the image kept.")
    top: int = Field(default=0, ge=0, description="Crop window top offset in pixels.")
    left: int = Field(default=0, ge=0, description="Crop window left offset in pixels.")