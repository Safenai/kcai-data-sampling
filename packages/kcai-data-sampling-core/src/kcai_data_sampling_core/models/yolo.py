"""A YOLO detector as a target model. Requires ``ultralytics`` (brings ``torch``)."""

from pathlib import Path

import numpy as np

from kcai_data_sampling_core.models.weights import weights_path


class YoloTarget:
    """``grad(xs)``: gradient of a loss w.r.t. the input batch, at the input's resolution.

    Loss = −(most confident class response), so ascending it lowers that
    confidence, untargeted, no annotation needed. The network runs at
    ``size``×``size``; the resize is differentiable.

    Args:
        weights: a checkpoint name resolved through :func:`weights_path`, or a path.
        size: square input size of the network.
    """

    def __init__(self, weights: str = "yolov8n.pt", size: int = 640):
        import torch
        from ultralytics import YOLO

        self._torch = torch
        path = Path(weights) if Path(weights).exists() else weights_path(weights)
        self.name = path.stem
        self.weights = path
        self.size = size
        self.net = YOLO(str(path)).model.eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    def grad(self, xs: np.ndarray) -> np.ndarray:
        """∂loss / ∂x for a batch, same shape as ``xs`` (``(B, C, H, W)`` float32 in [0, 1]).

        The loss is a sum over the batch of per-sample terms, so each row's
        gradient is its own.
        """
        torch = self._torch
        t = torch.from_numpy(np.ascontiguousarray(xs, dtype="float32")).requires_grad_(True)
        resized = torch.nn.functional.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
        prediction = self.net(resized)[0]  # (B, 4 + classes, anchors)
        loss = -prediction[:, 4:, :].amax(dim=(1, 2)).sum()
        loss.backward()
        return t.grad.numpy()
