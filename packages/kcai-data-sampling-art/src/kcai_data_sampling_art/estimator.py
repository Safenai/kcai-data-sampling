"""The adapter: a kcai target model seen as an ART estimator.

ART attacks take an *estimator*; kcai hands the transformation a
``TargetModel`` (``name`` + ``grad``). ``TargetEstimator`` bridges the two with
the smallest surface ART's white-box attacks check for: ``BaseEstimator`` plus
``LossGradientsMixin``, whose ``loss_gradient`` forwards to the target's
``grad``. The loss is the target's own, so the labels ART threads through are
unused. No model framework is imported: the gradient comes from the user's
model.
"""

from typing import Any

from art.estimators.estimator import BaseEstimator, LossGradientsMixin
import numpy as np


class TargetEstimator(LossGradientsMixin, BaseEstimator):
    """Expose a kcai ``TargetModel`` to ART's gradient-based attacks.

    The estimator works in kcai's normalized ``[0, 1]`` units, so
    ``clip_values`` is ``(0, 1)`` and ART clips the adversarial batch to it.
    It is not a ``ClassifierMixin``: ART then takes the caller's ``y`` as is
    instead of deriving labels from ``predict``.
    """

    estimator_params = BaseEstimator.estimator_params

    def __init__(self, target: Any, input_shape: tuple[int, ...]) -> None:
        """Wrap the target model.

        Args:
            target: The kcai target model; only its ``grad`` is used.
            input_shape: The shape of one sample, ``(H, W, C)``.
        """
        super().__init__(model=target, clip_values=(0.0, 1.0))
        self._input_shape = input_shape

    @property
    def input_shape(self) -> tuple[int, ...]:
        """The shape of one sample."""
        return self._input_shape

    def loss_gradient(self, x: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> np.ndarray:
        """Forward to the target's ``grad``; ``y`` is unused (the loss is the target's).

        Args:
            x: Batch normalized to ``[0, 1]``.
            y: Labels threaded through by ART; ignored.
            **kwargs: Unused.

        Returns:
            The target's gradient, same shape as ``x``.
        """
        del y, kwargs
        return np.asarray(self.model.grad(x))

    def predict(self, x: np.ndarray, **kwargs: Any) -> Any:
        """Not needed by FGSM when ``y`` is given; refused loudly otherwise."""
        raise NotImplementedError("TargetEstimator exposes gradients only: kcai targets have no predict")

    def fit(self, x: np.ndarray, y: Any, **kwargs: Any) -> None:
        """Targets are the user's, trained elsewhere: refused."""
        raise NotImplementedError("TargetEstimator does not train the user's target model")
