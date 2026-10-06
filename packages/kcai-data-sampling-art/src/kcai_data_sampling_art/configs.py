"""The ``art_fgsm`` transformation's configuration schema.

Same contract as the local ``fgsm``: ``epsilon`` is the L-infinity budget in
normalized ``[0, 1]`` pixel units, strictly positive and sweepable;
``target_model`` names a model from the ``models:`` section of the job config.
"""

from typing import Literal, Self

from kcai_data_sampling_core.models.config import TransformationConfig
from kcai_data_sampling_core.models.sweep import SweepConfig
from pydantic import Field, model_validator


class ArtFgsmTransformationConfig(TransformationConfig):
    """Configuration of the ``art_fgsm`` transformation.

    Attributes:
        target_model: Name of a target model from the job's ``models:``
            section; the CLI resolves it to an instance.
        epsilon: Adversarial budget in normalized ``[0, 1]`` pixel units,
            strictly positive, or a ``SweepConfig`` expanding it.
    """

    type: Literal["art_fgsm"] = "art_fgsm"
    target_model: str | None = Field(
        default=None,
        exclude=True,
        description="Name of a target model from the job's models: section; excluded"
        " from the resolved parameters (the base consumes it before validation).",
    )
    epsilon: float | SweepConfig = Field(
        description="Adversarial budget in normalized [0, 1] pixel units; a SweepConfig expands it."
    )

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce ``epsilon > 0`` on the value or the sweep.

        Returns:
            The validated config.
        """
        self._check_parameter_bounds("epsilon", minimum=0, exclusive_min=True)
        return self
