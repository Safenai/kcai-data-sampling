"""LaMa (Suvorov et al., WACV 2022, Samsung AI Lab, Apache-2.0) as a tool model.

The official *big-lama* checkpoint, TorchScript, ~200 MB. Requires ``torch``.
"""

import warnings

import numpy as np

from kcai_data_sampling.models.weights import fetch

URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"


class LamaTool:
    """``inpaint(xs, masks)``: the one thing the interface asks of this tool model.

    ``xs`` ``(B, C, H, W)`` float in [0, 1], ``masks`` ``(B, H, W)`` bool (True =
    erase and fill in). The network is fully convolutional and runs once on
    the batch, on a crop around the union of the masks (``margin`` pixels of
    context, sides padded to multiples of 8); only the masked pixels are
    written back, the rest of ``xs`` is returned untouched.
    """

    def __init__(self, weights: str = "big-lama.pt", margin: int = 256):
        import torch

        self._torch = torch
        self.weights = fetch(weights, URL)
        self.name = self.weights.stem
        self.margin = margin
        with warnings.catch_warnings():  # the checkpoint is TorchScript; torch >= 2.9 deprecates jit.load
            warnings.simplefilter("ignore", FutureWarning)
            self.net = torch.jit.load(str(self.weights), map_location="cpu").eval()

    def inpaint(self, xs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        torch = self._torch
        rows, cols = np.where(masks.any(axis=0))
        if rows.size == 0:
            return xs.copy()
        h, w = masks.shape[-2:]
        top, bottom = max(rows.min() - self.margin, 0), min(rows.max() + 1 + self.margin, h)
        left, right = max(cols.min() - self.margin, 0), min(cols.max() + 1 + self.margin, w)
        pad = ((0, 0), (0, (-(bottom - top)) % 8), (0, (-(right - left)) % 8))

        image = np.pad(xs[:, :, top:bottom, left:right], ((0, 0), *pad), mode="symmetric").astype("float32")
        hole = np.pad(masks[:, top:bottom, left:right], pad, mode="symmetric").astype("float32")
        with torch.inference_mode():
            out = self.net(torch.from_numpy(image), torch.from_numpy(hole)[:, None])
        filled = out[:, :, : bottom - top, : right - left].numpy()

        result = xs.copy()
        region, hole = result[:, :, top:bottom, left:right], masks[:, top:bottom, left:right]
        region[np.broadcast_to(hole[:, None], region.shape)] = filled[np.broadcast_to(hole[:, None], region.shape)]
        return result
