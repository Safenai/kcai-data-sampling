"""Image codec helpers and content-addressed artifact names."""

import hashlib
import io

from kcai_data_sampling_job.utils.images import artifact_name, decode_image, encode_image, png_bytes
import numpy as np
from PIL import Image
import pytest

WIDTH, HEIGHT = 8, 6


def _rgba_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(HEIGHT, WIDTH, 4), dtype=np.uint8)


def _write_png(path, frame: np.ndarray) -> None:
    Image.fromarray(frame).save(path)


def test_decode_roundtrips_a_png_to_rgba_bytes(tmp_path) -> None:
    """A PNG written by PIL decodes back to contiguous RGBA bytes."""
    frame = _rgba_frame()
    source = tmp_path / "in.png"
    _write_png(source, frame)
    height, width, raw = decode_image(source)
    assert (height, width) == (HEIGHT, WIDTH)
    assert bytes(frame.tobytes()) == raw


def test_decode_rejects_a_non_image_file(tmp_path) -> None:
    """A non-image file raises ValueError with the offending path."""
    source = tmp_path / "in.png"
    source.write_bytes(b"not an image")
    with pytest.raises(ValueError, match=r"cannot decode image .*in\.png"):
        decode_image(source)


def test_encode_writes_a_png_that_decodes_back(tmp_path) -> None:
    """``encode_image`` creates parent dirs and writes a decodable PNG."""
    frame = _rgba_frame()
    destination = tmp_path / "out" / "artifact.png"
    encode_image(destination, frame)
    height, width, raw = decode_image(destination)
    assert (height, width) == (HEIGHT, WIDTH)
    assert bytes(frame.tobytes()) == raw


def test_encode_rejects_a_non_uint8_array(tmp_path) -> None:
    """A float array raises without touching the disk."""
    destination = tmp_path / "artifact.png"
    with pytest.raises(ValueError, match=r"must be uint8"):
        encode_image(destination, _rgba_frame().astype(np.float64))


def test_encode_rejects_a_bad_shape(tmp_path) -> None:
    """A 1-D array (or wrong channel count) is refused with its real shape."""
    destination = tmp_path / "artifact.png"
    with pytest.raises(ValueError, match=r"expected \(H, W, 3\|4\) array, got \(48,\)"):
        encode_image(destination, np.zeros(WIDTH * HEIGHT, dtype=np.uint8))


def test_artifact_name_includes_the_content_hash_and_ids() -> None:
    """The stem carries selection, output id, and the sha1 content prefix."""
    frame = _rgba_frame()
    c6 = hashlib.sha1(bytes(frame.tobytes())).hexdigest()[:6]
    name = artifact_name("sel", "out-1", frame)
    assert name.endswith(".png")
    assert name.startswith("sel__out-1__")
    assert c6 in name


def test_artifact_name_is_content_addressed() -> None:
    """Identical pixels plus the same ids deduplicate to one name."""
    first = artifact_name("sel", "out", _rgba_frame(1))
    second = artifact_name("sel", "out", _rgba_frame(1))
    assert first == second
    assert first != artifact_name("sel", "out", _rgba_frame(2))


def test_png_bytes_encodes_without_writing() -> None:
    """``png_bytes`` returns PNG data decodable by PIL without touching disk."""
    frame = _rgba_frame()
    data = png_bytes(frame)
    decoded = Image.open(io.BytesIO(data))
    assert decoded.size == (WIDTH, HEIGHT)
    assert bytes(decoded.convert("RGBA").tobytes()) == bytes(frame.tobytes())
