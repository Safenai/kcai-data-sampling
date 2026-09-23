"""Image codec helpers and payload naming.

The RGBA-U8 pipeline: source files decode **once** to
raw RGBA bytes, flow through the engine as ``(B, H, W, 4)`` uint8 arrays, and
encode **once** to PNG at the outputs stage. Payload files are content-hashed
so identical generated images deduplicate.
"""

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def decode_image(path: str | Path) -> tuple[int, int, bytes]:
    """Decode an image file once into raw RGBA bytes.

    Args:
        path: Path to an image file.

    Returns:
        A ``(height, width, rgba_bytes)`` tuple where ``rgba_bytes`` is
        ``height * width * 4`` bytes of contiguous RGBA data (transparent
        pixels become alpha 0; opaque sources get alpha 255).

    Raises:
        ValueError: If the file is not a decodable image.
    """
    try:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            return rgba.height, rgba.width, rgba.tobytes()
    except Exception as exc:
        raise ValueError(f"cannot decode image {path}: {exc}") from exc


def encode_image(path: str | Path, array: np.ndarray) -> None:
    """Encode an HxW (3- or 4-channel) uint8 array as a PNG file.

    Args:
        path: Destination path (parent dirs are created).
        array: ``(H, W, C)`` uint8 array.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if array.dtype != np.uint8:
        raise ValueError("image arrays must be uint8")
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"expected (H, W, 3|4) array, got {array.shape}")
    Image.fromarray(np.ascontiguousarray(array)).save(path, format="PNG")


def quantize(x: np.ndarray) -> np.ndarray:
    """Quantize a float ``[0, 1]`` HWC array to uint8.

    Kept from the original codec for float paths (the RGBA-U8 pipeline skips
    quantization; end-to-end math stays in the 0..255 space).

    Args:
        x: Float array in ``[0, 1]``.

    Returns:
        A uint8 array formed by ``(x * 255 + 0.5).astype(np.uint8)``.
    """
    return np.clip(x, 0.0, 1.0) * 255.0 + 0.5 if x.dtype.kind == "f" else x.astype(np.uint8)


def artifact_name(selection_name: str, output_id: str, x: np.ndarray) -> str:
    """Build the content-addressed payload file name for a row.

    ``c6`` is a hash of the decoded payload bytes, so identical pixels plus
    the same output id name the same file (invariant artifact paths). The id
    already covers the recipe (parent, algorithm, settings, seed); ``c6`` only
    guards content collisions.

    Args:
        selection_name: Name of the selection the row belongs to.
        output_id: The output's derived ``id``.
        x: The output ``(H, W, C)`` uint8 pixel array.

    Returns:
        The ``.png`` artifact file stem to store under the payload directory.
    """
    c6 = hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:6]
    return f"{selection_name}__{output_id}__{c6}.png"


def png_bytes(x: np.ndarray) -> bytes:
    """Encode an HxW uint8 array to PNG bytes without touching the disk.

    Args:
        x: ``(H, W, 3|4)`` uint8 array.

    Returns:
        The PNG file bytes.
    """
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(x)).save(buffer, format="PNG")
    return buffer.getvalue()