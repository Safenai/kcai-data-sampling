"""LaMa (Suvorov et al., WACV 2022, Samsung AI Lab, Apache-2.0) as a tool model.

The official *big-lama* checkpoint, TorchScript, ~200 MB. Requires ``torch``.
"""

import warnings

import numpy as np

from kcai_data_sampling.models.weights import fetch

URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"


class LamaTool:
    """``inpaint(x, mask)``: the one thing the interface asks of this tool model.

    ``x`` CHW float in [0, 1], ``mask`` HW bool (True = erase and fill in).
    The network is fully convolutional and runs on a crop around the mask
    (``margin`` pixels of context, sides padded to multiples of 8); only the
    masked pixels are written back, the rest of ``x`` is returned untouched.
    A full 1928×1208 frame would take 4 GB; the crop takes a fraction.
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

    def inpaint(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        torch = self._torch
        rows, cols = np.where(mask)
        if rows.size == 0:
            return x.copy()
        h, w = mask.shape
        top, bottom = max(rows.min() - self.margin, 0), min(rows.max() + 1 + self.margin, h)
        left, right = max(cols.min() - self.margin, 0), min(cols.max() + 1 + self.margin, w)
        pad_h, pad_w = (-(bottom - top)) % 8, (-(right - left)) % 8

        image = np.pad(x[:, top:bottom, left:right], ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric").astype("float32")
        hole = np.pad(mask[top:bottom, left:right], ((0, pad_h), (0, pad_w)), mode="symmetric").astype("float32")
        with torch.inference_mode():
            out = self.net(torch.from_numpy(image)[None], torch.from_numpy(hole)[None, None])
        filled = out[0, :, : bottom - top, : right - left].numpy()

        result = x.copy()
        region = result[:, top:bottom, left:right]
        region[:, mask[top:bottom, left:right]] = filled[:, mask[top:bottom, left:right]]
        return result
