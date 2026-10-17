"""A JAX target model: any Hugging Face Flax image classifier.

Built like ``template_target.py``. Works with every Hugging Face image
classifier that has Flax weights (ViT, BEiT, ResNet, ...), chosen by model
id: the input size and normalization are read from the model's image
processor. Forward pass and gradient are pure JAX. The loss is the
cross-entropy of the model's own top-1 class, as in ``keras_classifier_target.py``.
``predict`` returns the class probabilities, for the black-box attacks.

``jax`` and ``transformers`` 4.x (5.x dropped the Flax classes) live in your
environment, never in kcai.

Config to point at it::

    models:
      vit:
        type: python
        path: examples/adapters/flax_classifier_target.py
        export: FlaxClassifierTarget
        weights: google/vit-base-patch16-224
"""

import numpy as np


class FlaxClassifierTarget:
    """A Hugging Face Flax image classifier seen by kcai as a target.

    Args:
        weights: Hugging Face model id with Flax weights, downloaded on first use.
    """

    def __init__(self, weights: str = "google/vit-base-patch16-224") -> None:
        import jax
        from transformers import AutoImageProcessor, FlaxAutoModelForImageClassification

        self.jax = jax
        self.model = FlaxAutoModelForImageClassification.from_pretrained(weights)
        self.name = weights.rsplit("/", 1)[-1]
        processor = AutoImageProcessor.from_pretrained(weights)
        size = getattr(processor, "crop_size", None) or processor.size
        self.input_size = (size["height"], size["width"])  # e.g. (224, 224)
        self.mean = np.asarray(processor.image_mean, dtype=np.float32)  # on [0, 1] pixel values
        self.std = np.asarray(processor.image_std, dtype=np.float32)

    def _inputs(self, rgb):
        """Resize to the model's input size (antialiased), normalize, channels first."""
        resized = self.jax.image.resize(rgb, (len(rgb), *self.input_size, 3), "bilinear")
        return ((resized - self.mean) / self.std).transpose(0, 3, 1, 2)

    def grad(self, xs: np.ndarray) -> np.ndarray:
        """Gradient of the top-1 cross-entropy with respect to ``xs``."""
        jax = self.jax

        def loss(rgb):
            # 2. Preprocess: the model's input size and normalization.
            inputs = self._inputs(rgb)
            # 3. Model: class logits (B, classes).
            logits = self.model(pixel_values=inputs).logits
            # 4. Loss: cross-entropy of the model's own top-1 class.
            top1 = jax.lax.stop_gradient(logits.argmax(axis=-1))
            return -jax.nn.log_softmax(logits)[jax.numpy.arange(len(rgb)), top1].sum()

        # 1. and 5. In JAX the input is the loss function's argument, and jax.grad differentiates it.
        dx = np.asarray(jax.grad(loss)(xs[..., :3].astype(np.float32)))
        # 6. Back to numpy, in the RGB planes; the alpha plane gets zero.
        result = np.zeros(xs.shape, dtype=np.float32)
        result[..., :3] = dx
        return result

    def predict(self, xs: np.ndarray) -> np.ndarray:
        """Class probabilities ``(B, classes)``."""
        logits = self.model(pixel_values=self._inputs(xs[..., :3].astype(np.float32))).logits
        return np.asarray(self.jax.nn.softmax(logits))
