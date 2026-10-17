"""A TensorFlow target model: a KerasHub object detector (D-FINE, RetinaNet).

Built like ``template_target.py``. Works with KerasHub's COCO detectors,
chosen by preset name (``dfine_nano_coco`` up to ``dfine_large_coco``, ...):
the input size is read from the preset's preprocessor. The loss is minus the
total confidence of the detections the model reports (class scores above
``conf``), as in ``torch_detector_target.py``: the attack lowers it, so it
suppresses detections.

The detector's ``predict`` decodes boxes and runs non-maximum suppression,
which has no gradient, so ``grad`` calls the network directly and reads its
raw class logits ``(B, boxes, classes)``. KerasHub's preprocessor crops the
image to a square; this adapter resizes the whole image instead, so every
pixel is seen (and attacked).

``tensorflow`` and ``keras-hub`` live in your environment, never in kcai.
``keras-hub`` requires full ``tensorflow`` through ``tensorflow-text``: next
to ``tensorflow-cpu``, install it with ``pip install --no-deps keras-hub
kagglehub kagglesdk``. With torch in the same process, import
``torch``/``torchvision`` before ``tensorflow``.

Config to point at it::

    models:
      dfine:
        type: python
        path: examples/adapters/keras_detector_target.py
        export: KerasDetectorTarget
        weights: dfine_nano_coco
        params: { conf: 0.5 }
"""

import numpy as np


class KerasDetectorTarget:
    """A KerasHub object detector seen by kcai as a target.

    Args:
        weights: KerasHub preset name, downloaded on first use.
        conf: Confidence above which a detection counts (and is attacked).
    """

    def __init__(self, weights: str = "dfine_nano_coco", conf: float = 0.5) -> None:
        import keras_hub
        import tensorflow as tf

        self.tf = tf
        self.model = keras_hub.models.ObjectDetector.from_preset(weights)
        self.name = weights
        converter = self.model.preprocessor.image_converter
        self.input_size = converter.image_size  # e.g. (640, 640)
        # The preset's scaling: the model expects pixel values in [0, 255] times scale, plus offset.
        self.scale = np.asarray(converter.scale if converter.scale is not None else 1.0, dtype=np.float32)
        self.offset = np.asarray(converter.offset if converter.offset is not None else 0.0, dtype=np.float32)
        self.conf = conf

    def grad(self, xs: np.ndarray) -> np.ndarray:
        """Gradient of minus the total detection confidence with respect to ``xs``."""
        tf = self.tf
        # 1. Tensor, tracking its gradient.
        rgb = tf.Variable(xs[..., :3].astype(np.float32))
        with tf.GradientTape() as tape:
            # 2. Preprocess: resize the whole image (antialiased, so every pixel gets a
            #    gradient), then the preset's scaling.
            inputs = tf.image.resize(rgb, self.input_size, antialias=True) * 255 * self.scale + self.offset
            # 3. Model: raw class logits (B, boxes, classes), before decoding and NMS
            #    ("logits" for D-FINE, "cls_logits" for RetinaNet).
            outputs = self.model(inputs, training=False)
            logits = outputs["logits"] if "logits" in outputs else outputs["cls_logits"]
            scores = tf.reduce_max(tf.sigmoid(logits), axis=-1)
            # 4. Loss: minus the confidence of the detections above the threshold.
            loss = -tf.reduce_sum(tf.where(scores > self.conf, scores, 0.0))
        # 5. Backpropagate to the input.
        dx = tape.gradient(loss, rgb).numpy()
        # 6. Back to numpy, in the RGB planes; the alpha plane gets zero.
        result = np.zeros(xs.shape, dtype=np.float32)
        result[..., :3] = dx
        return result
