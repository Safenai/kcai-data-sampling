"""A PyTorch target model: an ultralytics YOLO object detector (v8 and later).

Built like ``template_target.py``. Works with any ultralytics detection
checkpoint (``yolov8n.pt``, ``yolo11s.pt``, your own fine-tuned weights): the
stride and the number of classes are read from the model. The loss is minus
the total confidence of the detections the model reports (class scores above
``conf``): the attack lowers it, so it suppresses detections.

ultralytics' predictor (``YOLO.predict``) runs without gradients, so ``grad``
calls the torch network behind it directly. ``torch`` and ``ultralytics``
live in your environment, never in kcai; the weights are yours, e.g.
``yolov8n.pt`` fetched with ultralytics.

Config to point at it (see ``examples/config/walkthrough-adversarial.yaml``)::

    models:
      yolo:
        type: python
        path: examples/adapters/yolo_target.py
        export: YoloTarget
        weights: yolov8n.pt
        params: { conf: 0.25 }
"""

import numpy as np


class YoloTarget:
    """An ultralytics YOLO detector seen by kcai as a target.

    Args:
        weights: The ultralytics detection checkpoint, e.g. ``yolov8n.pt``.
        conf: Confidence above which a detection counts (and is attacked).
    """

    name = "yolo"

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25) -> None:
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.model = YOLO(weights).model.eval()  # the torch network behind the predictor
        self.stride = int(self.model.stride.max())  # image sides must be multiples of it
        self.num_classes = len(self.model.names)
        self.conf = conf

    def grad(self, xs: np.ndarray) -> np.ndarray:
        """Gradient of minus the total detection confidence with respect to ``xs``."""
        torch = self.torch
        # 1. Tensor (B, 3, H, W), tracking its gradient.
        rgb = torch.from_numpy(xs[..., :3].astype(np.float32)).permute(0, 3, 1, 2)
        rgb.requires_grad_(True)
        with torch.enable_grad():
            # 2. Preprocess: pad the bottom and right so both sides are multiples of the stride.
            height, width = rgb.shape[-2:]
            padded = torch.nn.functional.pad(rgb, (0, -width % self.stride, 0, -height % self.stride))
            # 3. Model: (B, 4 + classes, anchors), box coordinates then class scores per anchor.
            scores = self.model(padded)[0][:, -self.num_classes :].amax(dim=1)
            # 4. Loss: minus the confidence of the detections above the threshold.
            loss = -torch.where(scores > self.conf, scores, 0.0).sum()
            # 5. Backpropagate to the input.
            (dx,) = torch.autograd.grad(loss, rgb)
        # 6. Back to numpy, in the RGB planes; the alpha plane gets zero.
        result = np.zeros(xs.shape, dtype=np.float32)
        result[..., :3] = dx.permute(0, 2, 3, 1).numpy()
        return result
