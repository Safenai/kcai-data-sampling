"""Template: wrap your own model as a kcai target model.

Copy this file and fill in ``name``, ``__init__`` and the method(s) your
attacks need: ``grad`` for the gradient attacks (``fgsm``, ``art_fgsm``),
``predict`` for the black-box ones (``art_square``). Delete the one you do not
use. See ``docs/target-models.md``. Runnable examples:
``yolo_target.py`` (PyTorch), ``keras_target.py`` (TensorFlow) and
``flax_target.py`` (JAX).
"""

import numpy as np


class MyTarget:
    """Your model, seen by kcai as a target."""

    # A readable name, recorded in the ledger's ``target_model`` column.
    name = "my_model"

    def __init__(self, weights: str | None = None) -> None:
        """Load your model.

        The job builds the adapter as ``MyTarget(weights=<models.weights>,
        **<models.params>)``: add any argument you want to set from the YAML.

        Args:
            weights: Path to your checkpoint, from the ``weights`` key.
        """
        raise NotImplementedError("load your model here, e.g. self.model = ...")

    def grad(self, xs: np.ndarray) -> np.ndarray:
        """Gradient of your loss with respect to ``xs``.

        Args:
            xs: Float images ``(B, H, W, C)`` in ``[0, 1]`` (``C = 4``: RGBA).

        Returns:
            A finite numpy array with exactly the shape of ``xs``. The attack
            moves pixels in the direction that increases the loss.
        """
        # 1. Convert xs[..., :3] to your framework's tensor and track its gradient.
        # 2. Preprocess with framework ops (resize with antialiasing, normalize).
        # 3. Run the model in evaluation mode.
        # 4. Compute the scalar loss the attack should increase.
        # 5. Backpropagate to the input tensor.
        # 6. Write the gradient into the RGB planes of np.zeros(xs.shape) and return it.
        raise NotImplementedError("compute the gradient of your loss here")

    def predict(self, xs: np.ndarray) -> np.ndarray:
        """Class scores, for the black-box attacks (``art_square``).

        Args:
            xs: Float images ``(B, H, W, C)`` in ``[0, 1]`` (``C = 4``: RGBA).

        Returns:
            A finite numpy array ``(B, n_classes)``: probabilities or logits.
        """
        # Same input steps as grad (tensor, preprocess, model in evaluation mode),
        # without gradient tracking; return the class scores as numpy.
        raise NotImplementedError("return your model's class scores here")
