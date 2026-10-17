"""A PyTorch target model: any torchvision image classifier.

Built like ``template_target.py``. Works with every classifier in
``torchvision.models`` (``resnet50``, ``mobilenet_v3_large``, ``vit_b_16``,
...), chosen by name: the input size and normalization are read from the
weights' own preprocessing. ``grad`` uses ``torch.autograd``; the loss is the
cross-entropy of the model's own top-1 class, as in ``keras_classifier_target.py``.
``predict`` returns the class probabilities, for the black-box attacks.

``torch`` and ``torchvision`` live in your environment, never in kcai. With
tensorflow in the same process, import ``torch``/``torchvision`` first: the
reverse order crashes with tensorflow-cpu 2.21 and torchvision 0.29.

Config to point at it::

    models:
      resnet:
        type: python
        path: examples/adapters/torch_classifier_target.py
        export: TorchClassifierTarget
        weights: DEFAULT
        params: { model: resnet50 }
"""

import numpy as np


class TorchClassifierTarget:
    """A torchvision classifier seen by kcai as a target.

    Args:
        model: Name of the model in ``torchvision.models``, e.g. ``"resnet50"``.
        weights: torchvision weights name, e.g. ``"DEFAULT"`` or
            ``"IMAGENET1K_V1"``; downloaded on first use.
    """

    def __init__(self, model: str = "resnet50", weights: str = "DEFAULT") -> None:
        import torch
        import torchvision

        self.torch = torch
        self.model = torchvision.models.get_model(model, weights=weights).eval()
        self.name = model
        # The preprocessing torchvision ships with the weights (expects pixel values in [0, 1]).
        preprocess = torchvision.models.get_model_weights(model)[weights].transforms()
        side = preprocess.crop_size  # a square crop, e.g. [224]
        self.input_size = (side[0], side[-1])  # (224, 224)
        self.mean = torch.tensor(preprocess.mean).view(1, 3, 1, 1)
        self.std = torch.tensor(preprocess.std).view(1, 3, 1, 1)

    def _logits(self, rgb):
        """Resize (antialiased, so every pixel gets a gradient), normalize, run the model."""
        resized = self.torch.nn.functional.interpolate(
            rgb, size=self.input_size, mode="bilinear", antialias=True, align_corners=False
        )
        return self.model((resized - self.mean) / self.std)

    def grad(self, xs: np.ndarray) -> np.ndarray:
        """Gradient of the top-1 cross-entropy with respect to ``xs``."""
        torch = self.torch
        # 1. Tensor (B, 3, H, W), tracking its gradient.
        rgb = torch.from_numpy(xs[..., :3].astype(np.float32)).permute(0, 3, 1, 2)
        rgb.requires_grad_(True)
        with torch.enable_grad():
            # 2-3. Preprocess and model: logits (B, classes).
            logits = self._logits(rgb)
            # 4. Loss: cross-entropy of the model's own top-1 class.
            top1 = logits.argmax(dim=-1).detach()
            loss = torch.nn.functional.cross_entropy(logits, top1, reduction="sum")
            # 5. Backpropagate to the input.
            (dx,) = torch.autograd.grad(loss, rgb)
        # 6. Back to numpy, in the RGB planes; the alpha plane gets zero.
        result = np.zeros(xs.shape, dtype=np.float32)
        result[..., :3] = dx.permute(0, 2, 3, 1).numpy()
        return result

    def predict(self, xs: np.ndarray) -> np.ndarray:
        """Class probabilities ``(B, classes)``."""
        torch = self.torch
        rgb = torch.from_numpy(xs[..., :3].astype(np.float32)).permute(0, 3, 1, 2)
        with torch.no_grad():
            return torch.softmax(self._logits(rgb), dim=-1).numpy()
