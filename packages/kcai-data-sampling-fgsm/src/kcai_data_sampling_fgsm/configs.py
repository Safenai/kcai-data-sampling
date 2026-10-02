"""The ``fgsm`` transformation's configuration schema.

The schema subclasses the core base ``TransformationConfig`` and pins ``type``
to the algorithm's literal, so the registry-resolved validator in ``JobConfig``
picks it by type. ``epsilon`` — the adversarial budget — is required, in
normalized ``[0, 1]`` pixel units, strictly positive, and sweepable (its
interval must stay ``> 0``). ``target_model`` names a model from the
``models:`` section of the job config; the CLI resolves the name to an
instance.
"""

from typing import Literal, Self

from kcai_data_sampling_core.models.config import TransformationConfig
from kcai_data_sampling_core.models.sweep import SweepConfig
from pydantic import Field, model_validator


class FgsmTransformationConfig(TransformationConfig):
    """Configuration of the ``fgsm`` transformation.

    Attributes:
        target_model: Name of a target model from the job's ``models:``
            section; the CLI resolves it to an instance. Optional here because
            the transformation base consumes it before re-validating the
            remaining parameters against this schema; a model-role
            transformation without one is refused loudly at construction (the
            slot check).
        epsilon: Adversarial budget in normalized ``[0, 1]`` pixel units,
            strictly positive, or a ``SweepConfig`` expanding it (its interval
            must stay ``> 0``).
    """

    type: Literal["fgsm"] = "fgsm"
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
        """Enforce the epsilon budget's range on the value or the sweep.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``epsilon`` is not strictly positive, whether given
                directly or as a sweep interval.
        """
        self._check_parameter_bounds("epsilon", minimum=0, exclusive_min=True)
        return self
