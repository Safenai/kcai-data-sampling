"""The adapters: a kcai target model seen as an ART estimator.

ART attacks take an *estimator*; kcai hands the transformation a target model.
Two bridges, one per kind of access the attacks need:

- ``TargetEstimator`` (gradient attacks such as FGSM): ``BaseEstimator`` plus
  ``LossGradientsMixin``, whose ``loss_gradient`` forwards to the target's
  ``grad``. The loss is the target's own, so the labels ART threads through are
  unused.
- ``TargetClassifier`` (black-box attacks such as Square): a classifier whose
  ``predict`` forwards to the target's ``predict`` (class scores). No gradient
  is needed.

They stay separate on purpose: ART's gradient attacks call ``predict`` as soon
as the estimator is a classifier, which a grad-only target cannot answer. No
model framework is imported: gradients and scores come from the user's model.
"""

from typing import Any

from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
import numpy as np


class TargetEstimator(LossGradientsMixin, BaseEstimator):  # type: ignore[misc]  # ART ships no type hints
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


class TargetClassifier(ClassifierMixin, NeuralNetworkMixin, BaseEstimator):  # type: ignore[misc]  # ART ships no type hints
    """Expose a kcai target model's ``predict`` to ART's black-box attacks.

    Inputs stay in kcai's layout and units: ``(B, H, W, C)`` (channels last)
    in ``[0, 1]``, so ``clip_values`` is ``(0, 1)``. The number of classes is
    read from the first ``predict`` answer.
    """

    estimator_params = BaseEstimator.estimator_params + NeuralNetworkMixin.estimator_params

    def __init__(self, target: Any, input_shape: tuple[int, ...]) -> None:
        """Wrap the target model.

        Args:
            target: The kcai target model; only its ``predict`` is used.
            input_shape: The shape of one sample, ``(H, W, C)``.
        """
        super().__init__(model=target, clip_values=(0.0, 1.0), channels_first=False)
        self._input_shape = input_shape

    @property
    def input_shape(self) -> tuple[int, ...]:
        """The shape of one sample."""
        return self._input_shape

    @property
    def nb_classes(self) -> int:
        """Number of classes, known once ``predict`` has answered."""
        return int(self._nb_classes)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs: Any) -> np.ndarray:
        """Forward to the target's ``predict`` and check its answer.

        Args:
            x: Batch normalized to ``[0, 1]``.
            batch_size: Unused: the target receives the batch ART passes.
            **kwargs: Unused.

        Returns:
            Class scores ``(B, n_classes)``.

        Raises:
            ValueError: If the answer is not a finite ``(B, n_classes)`` array
                with at least two classes.
        """
        del batch_size, kwargs
        scores = self.model.predict(x)
        if not isinstance(scores, np.ndarray) or scores.ndim != 2 or len(scores) != len(x) or scores.shape[1] < 2:
            raise ValueError(
                f"{self.model.name}.predict returned {type(scores).__name__}{getattr(scores, 'shape', '')}, "
                f"expected class scores of shape ({len(x)}, n_classes) with n_classes >= 2"
            )
        if not np.isfinite(scores).all():
            raise ValueError(f"{self.model.name}.predict returned non-finite values")
        self._nb_classes = scores.shape[1]
        return scores

    def fit(self, x: np.ndarray, y: Any, **kwargs: Any) -> None:
        """Targets are the user's, trained elsewhere: refused."""
        raise NotImplementedError("TargetClassifier does not train the user's target model")

    def get_activations(self, x: np.ndarray, layer: int | str, batch_size: int, framework: bool = False) -> np.ndarray:
        """kcai targets expose class scores only: refused."""
        raise NotImplementedError("TargetClassifier exposes class scores only: kcai targets have no activations")
