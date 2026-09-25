"""Semantics of the ``crop_resize`` transformation.

The crop keeps the ``top:top+fraction*H`` by ``left:left+fraction*W`` window
and nearest-neighbor resizes it back to the sample shape (the identity and
window math): ``fraction=1`` must be byte-exact identity, values stay uint8 in
``[0, 255]``, and a window that leaves the image is refused. The class comes
from ``kcai_data_sampling_images.api.transformations``.
"""

import numpy as np
import pytest

from kcai_data_sampling_images.api.transformations.crop_resize import CropResize


def _out_of(batch, fraction: float, top: int = 0, left: int = 0) -> np.ndarray:
    """Transform one batch with ``CropResize`` and return the stacked outputs."""
    transform = CropResize({"fraction": fraction, "top": top, "left": left})
    outputs = transform.transform(batch, batch.value_range)
    return np.stack([o.x for o in outputs])


def test_fraction_one_is_byte_exact_identity(synthetic_batch) -> None:
    """``fraction=1.0`` replays the source bytes exactly.

    A full-window crop resizes back through an identity index map, so the
    contract is ``output == input`` byte-exact — no re-encode drift anywhere.
    """
    assert _out_of(synthetic_batch, 1.0).tobytes() == synthetic_batch.data.tobytes()


def test_fraction_half_keeps_the_top_left_window(synthetic_batch) -> None:
    """A 0.5 crop is the nearest-neighbor blow-up of the top-left 50%.

    The spec map (window at ``top,left``, then index replication
    ``src[y//2, x//2]``-style) is asserted pixel-by-pixel on every coordinate,
    so window placement and resize are pinned together, not just the class's
    output equality to itself.
    """
    transformed = _out_of(synthetic_batch, 0.5, top=8, left=8)
    frame = synthetic_batch.images[2]
    h, w = frame.shape[:2]
    window_h, window_w = h // 2, w // 2
    for y in range(h):
        for x in range(w):
            src_y = 8 + (y * window_h) // h
            src_x = 8 + (x * window_w) // w
            assert np.array_equal(transformed[2, y, x], frame[src_y, src_x])


def test_crop_on_the_bar_hardens_to_one_color(synthetic_batch) -> None:
    """Crop of the two-tone bar's left half is uniformly its left color.

    Geometry check: a 0.5 top-left crop of the bar frame contains only the
    half of the bar with the left color, so resizing back yields a flat image
    of exactly that color.
    """
    transformed = _out_of(synthetic_batch, 0.5)
    left_color = synthetic_batch.images[0][0, 0]
    assert (transformed[0] == left_color).all()


def test_crop_preserves_shape_space_and_value_range(synthetic_batch) -> None:
    """The output is the same ``(H, W, C)`` uint8 sample as the input.

    Even though the crop discards information, the resize back keeps the
    sample's space (shape and range) intact — the unary contract that leaves
    ``δ`` defined.
    """
    transformed = _out_of(synthetic_batch, 0.5)
    assert transformed.shape == synthetic_batch.data.shape
    assert transformed.dtype == np.uint8
    assert transformed.min() >= 0 and transformed.max() <= 255


def test_crop_is_not_reversible(synthetic_batch) -> None:
    """The crop discards information: it is declared non-reversible.

    A 0.5 crop cannot map back to the source (the discarded region is gone), so
    ``reversible`` is False on the row — a structural declaration, mirrored by
    the fact that cropping twice differs from the source.
    """
    transform = CropResize({"fraction": 0.5})
    declaration = transform.describe()
    assert declaration == {
        "algorithm": "crop_resize",
        "family": "procedural",
        "arity": "unary",
        "reversible": False,
        "params": {"fraction": 0.5, "top": 0, "left": 0},
        "seed": None,
        "tool_model": None,
        "target_model": None,
    }
    once = _out_of(synthetic_batch, 0.5)
    assert not np.array_equal(once, synthetic_batch.data)


def test_window_leaving_the_image_is_refused(synthetic_batch) -> None:
    """A crop window that crosses the image boundary is a config error.

    ``top``/``left`` too large for the computed window must fail loudly (a
    value-range style check before any indexing), not silently crop short.
    """
    with pytest.raises(ValueError, match="leaves the"):
        _out_of(synthetic_batch, 0.5, top=24, left=0)


def test_crop_is_independent_of_the_batch_split(synthetic_batch) -> None:
    """Row-by-row cropping matches the full-batch run byte-for-byte.

    Like the flip, the crop is a per-row pixel map: the batch split never
    changes a row's output.
    """
    whole = _out_of(synthetic_batch, 0.5)
    for i in range(len(synthetic_batch)):
        row = synthetic_batch.row(i)
        assert np.array_equal(_out_of(row, 0.5)[0], whole[i])