"""LaMa (Suvorov et al., WACV 2022) as a tool model.

Wraps the official *big-lama* TorchScript checkpoint (~200 MB, downloaded into
the weight cache on first use) behind the ``inpaint(xs, masks)`` tool-model
contract. ``torch`` is imported lazily so that importing this package never
requires it; ``big-lama`` itself is a weights file fetched at run time, not a
dependency.
"""

import warnings

import numpy as np

from kcai_data_sampling_images_lama.api.models.weights import fetch

URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"


class LamaTool:
    """The ``inpaint`` tool model backed by LaMa.

    Attributes:
        name: The checkpoint's file stem (``"big-lama"``), recorded on every
            output row.
        weights: The local path of the checkpoint in the weight cache.
        margin: Pixels of context cropped around the mask union for the network.
        channels: Image planes the network consumes (``3`` = RGB). The alpha
            plane of an RGBA batch is then never sent to the network and is
            returned untouched.
    """

    channels = 3

    def __init__(self, weights: str = "big-lama.pt", margin: int = 256):
        """Load the LaMa checkpoint.

        Args:
            weights: The checkpoint's file name in the weight cache; downloaded
                from the upstream release on first use.
            margin: Pixels of context cropped around the mask union for the
                network (sides are padded to multiples of 8).

        Raises:
            ImportError: If ``torch`` is not installed.
        """
        import torch

        self._torch = torch
        self.weights = fetch(weights, URL)
        self.name = self.weights.stem
        self.margin = margin
        with warnings.catch_warnings():  # TorchScript checkpoint; torch >= 2.9 deprecates jit.load
            warnings.simplefilter("ignore", FutureWarning)
            self.net = torch.jit.load(str(self.weights), map_location="cpu").eval()

    def inpaint(self, xs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Fill every masked pixel with network output, leave the rest untouched.

        The ``(B, H, W, 4)`` uint8 RGBA batch is converted to the network's
        NCHW float input (the ``channels`` first planes, in ``[0, 1]``) and back.
        The network runs once, on a crop around the union of the masks plus
        ``margin`` pixels of context (sides padded to multiples of 8); only the
        masked pixels of the selected planes are rewritten.

        Args:
            xs: Batch of image arrays shaped ``(B, H, W, 4)``, uint8 RGBA.
            masks: One boolean mask per sample ``(B, H, W)``; True = erase and fill in.

        Returns:
            A batch of the same shape, uint8 in ``[0, 255]``. Unmasked pixels
            and (when ``channels`` is 3) the whole alpha plane are exactly as
            in ``xs``.

        Raises:
            ValueError: If ``xs`` does not carry at least ``channels`` planes.
        """
        if xs.ndim != 4 or xs.shape[-1] < self.channels:
            raise ValueError(f"LamaTool.inpaint expects a (B, H, W, 4) uint8 RGBA batch, got shape {xs.shape}")
        rows, cols = np.where(masks.any(axis=0))
        if rows.size == 0:
            return xs.copy()
        h, w = masks.shape[-2:]
        top, bottom = max(rows.min() - self.margin, 0), min(rows.max() + 1 + self.margin, h)
        left, right = max(cols.min() - self.margin, 0), min(cols.max() + 1 + self.margin, w)
        pad_h, pad_w = (-(bottom - top)) % 8, (-(right - left)) % 8

        image = np.pad(
            xs[:, top:bottom, left:right, : self.channels],
            ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
            mode="symmetric",
        )
        image = image.astype("float32").transpose(0, 3, 1, 2) / 255.0
        hole = np.pad(
            masks[:, top:bottom, left:right],
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="symmetric",
        ).astype("float32")
        torch = self._torch
        with torch.inference_mode():
            out = self.net(torch.from_numpy(image), torch.from_numpy(hole)[:, None])
        filled = out[:, :, : bottom - top, : right - left].numpy()

        result = xs.copy()
        region = result[:, top:bottom, left:right, : self.channels]
        hole_pixels = np.broadcast_to(masks[:, top:bottom, left:right, None], region.shape)
        filled_pixels = np.transpose(filled, (0, 2, 3, 1))[hole_pixels]
        region[hole_pixels] = np.clip(filled_pixels * 255.0, 0.0, 255.0).astype(np.uint8)
        return result
