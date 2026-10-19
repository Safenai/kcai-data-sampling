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


class ArtSquareTransformationConfig(TransformationConfig):
    """Configuration of the ``art_square`` transformation.

    Attributes:
        target_model: Name of a target model from the job's ``models:``
            section; it must expose ``predict``.
        epsilon: L-infinity budget in normalized ``[0, 1]`` pixel units,
            strictly positive, or a ``SweepConfig`` expanding it.
        max_iter: Query budget: at most this many ``predict`` calls per image
            after the first ones (the attack stops early once the prediction
            changes).
        p_init: Initial fraction of the image covered by each random square,
            in ``(0, 1]``. Defaults to 0.05, the L-infinity value of the Square
            attack paper (ART's own default, 0.8, makes the first square larger
            than any landscape image is tall).
    """

    type: Literal["art_square"] = "art_square"
    target_model: str | None = Field(
        default=None,
        exclude=True,
        description="Name of a target model from the job's models: section; excluded"
        " from the resolved parameters (the base consumes it before validation).",
    )
    epsilon: float | SweepConfig = Field(
        description="Adversarial budget in normalized [0, 1] pixel units; a SweepConfig expands it."
    )
    max_iter: int = Field(default=100, gt=0, description="Query budget per image.")
    p_init: float = Field(default=0.05, gt=0, le=1, description="Initial fraction of the image per square.")

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce ``epsilon > 0`` on the value or the sweep.

        Returns:
            The validated config.
        """
        self._check_parameter_bounds("epsilon", minimum=0, exclusive_min=True)
        return self


_TARGET_MODEL_FIELD = Field(
    default=None,
    exclude=True,
    description="Name of a target model from the job's models: section; excluded"
    " from the resolved parameters (the base consumes it before validation).",
)


class ArtPgdTransformationConfig(TransformationConfig):
    """Configuration of the ``art_pgd`` transformation.

    Attributes:
        target_model: Name of a target model from the job's ``models:``
            section; it must expose ``grad``.
        epsilon: Budget in normalized ``[0, 1]`` pixel units, strictly
            positive, or a ``SweepConfig`` expanding it. For ``norm: 2`` it
            bounds the L2 norm of the whole perturbation (all pixels and
            channels), so it is much larger than an L-infinity budget.
        norm: ``inf`` (each pixel moves by at most ``epsilon``) or ``2``.
        max_iter: Number of gradient steps.
        step_fraction: Size of each step, as a fraction of ``epsilon``. The
            default ``0.25`` with ``max_iter: 10`` walks ``2.5 * epsilon`` in
            total, the usual PGD choice.
    """

    type: Literal["art_pgd"] = "art_pgd"
    target_model: str | None = _TARGET_MODEL_FIELD
    epsilon: float | SweepConfig = Field(
        description="Adversarial budget in normalized [0, 1] pixel units; a SweepConfig expands it."
    )
    norm: Literal["inf", 2] = Field(default="inf", description="Norm of the budget: inf or 2.")
    max_iter: int = Field(default=10, gt=0, description="Number of gradient steps.")
    step_fraction: float = Field(default=0.25, gt=0, le=1, description="Step size, as a fraction of epsilon.")

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce ``epsilon > 0`` on the value or the sweep.

        Returns:
            The validated config.
        """
        self._check_parameter_bounds("epsilon", minimum=0, exclusive_min=True)
        return self


class ArtMimTransformationConfig(ArtPgdTransformationConfig):
    """Configuration of the ``art_mim`` transformation: ``art_pgd``'s, plus ``decay``.

    Attributes:
        decay: Momentum decay factor: each step follows the gradient plus
            ``decay`` times the accumulated previous directions.
    """

    type: Literal["art_mim"] = "art_mim"  # type: ignore[assignment]  # narrows the parent's literal
    decay: float = Field(default=1.0, ge=0, description="Momentum decay factor.")


class ArtSimbaTransformationConfig(TransformationConfig):
    """Configuration of the ``art_simba`` transformation.

    Attributes:
        target_model: Name of a target model from the job's ``models:``
            section; its ``predict`` must return probabilities (rows summing
            to 1), not logits.
        epsilon: Change applied to one pixel value per query, in normalized
            ``[0, 1]`` units; each value changes at most once, so it is also
            the L-infinity budget. Strictly positive, or a ``SweepConfig``.
        max_iter: Query budget: the number of pixel values tried per image
            (the attack stops early once the prediction changes).
    """

    type: Literal["art_simba"] = "art_simba"
    target_model: str | None = _TARGET_MODEL_FIELD
    epsilon: float | SweepConfig = Field(
        description="Per-pixel change and L-infinity budget, in normalized [0, 1] units; a SweepConfig expands it."
    )
    max_iter: int = Field(default=1000, gt=0, description="Query budget per image.")

    @model_validator(mode="after")
    def _parameter_bounds(self) -> Self:
        """Enforce ``epsilon > 0`` on the value or the sweep.

        Returns:
            The validated config.
        """
        self._check_parameter_bounds("epsilon", minimum=0, exclusive_min=True)
        return self


class ArtHopSkipJumpTransformationConfig(TransformationConfig):
    """Configuration of the ``art_hopskipjump`` transformation.

    HopSkipJump has no budget: it starts from a misclassified noise image and
    walks back toward the original, minimizing the distance in ``norm``. The
    change is therefore not bounded; check it on the output rows.

    Attributes:
        target_model: Name of a target model from the job's ``models:``
            section; only the top class of its ``predict`` is used.
        norm: Distance minimized: ``2`` or ``inf``.
        max_iter: Number of iterations (gradient estimate, step, boundary search).
        max_eval: Most queries for one gradient estimate. Each query is a
            noisy copy of the image held in memory: lower it for large images.
        init_eval: Queries for the first gradient estimate (grows with the iterations).
        init_size: Noise images tried to find a misclassified starting point;
            if none is, the image is returned unchanged.
    """

    type: Literal["art_hopskipjump"] = "art_hopskipjump"
    target_model: str | None = _TARGET_MODEL_FIELD
    norm: Literal["inf", 2] = Field(default=2, description="Distance minimized: 2 or inf.")
    max_iter: int = Field(default=10, ge=0, description="Number of iterations.")
    max_eval: int = Field(default=100, gt=0, description="Most queries per gradient estimate.")
    init_eval: int = Field(default=10, gt=0, description="Queries for the first gradient estimate.")
    init_size: int = Field(default=100, gt=0, description="Noise images tried for a starting point.")

    @model_validator(mode="after")
    def _eval_bounds(self) -> Self:
        """Enforce ``init_eval <= max_eval`` (ART refuses otherwise, mid-run).

        Returns:
            The validated config.

        Raises:
            ValueError: If ``init_eval`` exceeds ``max_eval``.
        """
        if self.init_eval > self.max_eval:
            raise ValueError(f"art_hopskipjump: init_eval={self.init_eval} must not exceed max_eval={self.max_eval}")
        return self
