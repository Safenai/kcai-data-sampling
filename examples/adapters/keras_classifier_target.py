"""A TensorFlow target model: any Keras Applications image classifier.

Built like ``template_target.py``. Works with every classifier in
``keras.applications`` (``MobileNetV2``, ``ResNet50``, ``EfficientNetB0``,
...), chosen by name: the input size is read from the model, and the
preprocessing is Keras's own ``preprocess_input`` for that model family.
``grad`` uses ``tf.GradientTape``; the loss is the cross-entropy of the
model's own top-1 class, so the attack pushes the prediction away from what
the model currently sees (no label needed). ``predict`` returns the class
probabilities, for the black-box attacks.

``tensorflow`` lives in your environment, never in kcai. With torch in the
same process, import ``torch``/``torchvision`` before ``tensorflow``: the
reverse order crashes with tensorflow-cpu 2.21 and torchvision 0.29.

Config to point at it::

    models:
      mobilenet:
        type: python
        path: examples/adapters/keras_classifier_target.py
        export: KerasClassifierTarget
        weights: imagenet
        params: { model: MobileNetV2 }
"""

import importlib

import numpy as np


class KerasClassifierTarget:
    """A Keras Applications classifier seen by kcai as a target.

    Args:
        model: Name of the model in ``keras.applications``, e.g. ``"MobileNetV2"``.
        weights: Keras weights; ``"imagenet"`` is downloaded on first use.
    """

    def __init__(self, model: str = "MobileNetV2", weights: str = "imagenet") -> None:
        import tensorflow as tf

        self.tf = tf
        build = getattr(tf.keras.applications, model)
        self.model = build(weights=weights)
        self.name = self.model.name
        self.input_size = self.model.input_shape[1:3]  # e.g. (224, 224)
        # The preprocessing Keras ships next to the model (expects pixel values in [0, 255]).
        self.preprocess_input = importlib.import_module(build.__module__).preprocess_input

    def grad(self, xs: np.ndarray) -> np.ndarray:
        """Gradient of the top-1 cross-entropy with respect to ``xs``."""
        tf = self.tf
        # 1. Tensor, tracking its gradient.
        rgb = tf.Variable(xs[..., :3].astype(np.float32))
        with tf.GradientTape() as tape:
            # 2. Preprocess: resize to the model's input size (antialiased, so every pixel
            #    gets a gradient), then the model's own preprocessing.
            inputs = self.preprocess_input(tf.image.resize(rgb, self.input_size, antialias=True) * 255)
            # 3. Model: class probabilities (B, classes).
            probs = self.model(inputs, training=False)
            # 4. Loss: cross-entropy of the model's own top-1 class.
            top1 = tf.stop_gradient(tf.argmax(probs, axis=-1))
            loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(top1, probs))
        # 5. Backpropagate to the input.
        dx = tape.gradient(loss, rgb).numpy()
        # 6. Back to numpy, in the RGB planes; the alpha plane gets zero.
        result = np.zeros(xs.shape, dtype=np.float32)
        result[..., :3] = dx
        return result

    def predict(self, xs: np.ndarray) -> np.ndarray:
        """Class probabilities ``(B, classes)``."""
        rgb = xs[..., :3].astype(np.float32)
        inputs = self.preprocess_input(self.tf.image.resize(rgb, self.input_size, antialias=True) * 255)
        return self.model(inputs, training=False).numpy()
