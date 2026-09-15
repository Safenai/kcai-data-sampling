"""A YOLO detector as a target model. Requires ``ultralytics`` (brings ``torch``)."""

import os
from pathlib import Path

import numpy as np


def weights_path(name: str = "yolov8n.pt") -> Path:
    """``$KCAI_WEIGHTS_DIR`` or ``~/.cache/kcai-data-sampling/weights``: never
    the working directory, where ultralytics would otherwise save."""
    directory = Path(os.environ.get("KCAI_WEIGHTS_DIR", Path.home() / ".cache" / "kcai-data-sampling" / "weights"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


class YoloTarget:
    """``grad(x)``: gradient of a loss w.r.t. the input, at the input's resolution.

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

    def grad(self, x: np.ndarray) -> np.ndarray:
        """∂loss / ∂x, same shape as ``x`` (CHW float32 in [0, 1])."""
        torch = self._torch
        t = torch.from_numpy(np.ascontiguousarray(x, dtype="float32"))[None].requires_grad_(True)
        resized = torch.nn.functional.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
        prediction = self.net(resized)[0]  # (1, 4 + classes, anchors)
        loss = -prediction[:, 4:, :].amax()
        loss.backward()
        return t.grad[0].numpy()
