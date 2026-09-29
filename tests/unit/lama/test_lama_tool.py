"""The opt-in ``LamaTool`` behaviors on the real checkpoint.

This module runs only in the ``test-lama`` nox env; elsewhere it is skipped
visible, exactly like the other ``-lama`` test modules. It pins the *model
adapter* contract — the class-level declarations, the empty-mask fast path,
real-torch determinism, and the ``cli._build_models`` wiring from a bare
``ModelRefConfig`` — leaving the transformation-level coverage to
``tests/unit/images/test_inpaint.py``.

The crop-and-pad-to-multiples-of-8 path is exercised by construction on the
32x32 frames here (the default 256-pixel margin clamps the crop to the whole
frame) and explicitly with a non-multiple-of-8 36x36 frame in
``test_crop_and_pad_to_multiples_of_8``, so both the zero-pad and the
nonzero-pad branches are covered.
"""

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_job import cli
import numpy as np
import pytest
from tests.fixtures.data import REGION
from tests.utils.configs import build_config, build_loader

pytest.importorskip("kcai_data_sampling_images_lama.api.models.lama")

pytestmark = pytest.mark.lama

from kcai_data_sampling_images_lama.api.models.lama import LamaTool  # noqa: E402


@pytest.fixture(scope="module")
def lama_tool(cached_big_lama) -> LamaTool:
    """One real ``LamaTool`` shared by this module's tests.

    Returns:
        A ``LamaTool`` over the session-fetched ``big-lama.pt``.
    """
    return LamaTool(weights=str(cached_big_lama))


def _region_masks(batch) -> np.ndarray:
    """One boolean ``(B, H, W)`` mask per row, True on the REGION window.

    Args:
        batch: An array-holding batch with ``data.shape == (B, H, W, C)``.

    Returns:
        The mask batch.
    """
    masks = np.zeros(batch.data.shape[:3], dtype=bool)
    top, left, height, width = REGION.values()
    masks[:, top : top + height, left : left + width] = True
    return masks


def test_lama_tool_class_declarations(lama_tool) -> None:
    """The adapter's contract surface: 3 planes, name from the stem, margin 256.

    ``channels == 3`` is what keeps the alpha plane of an RGBA batch out of
    the network; the recorded ``name`` and the default ``margin`` are the
    production values the transformations registry and the ``models:`` section
    rely on.
    """
    assert lama_tool.channels == 3
    assert lama_tool.name == "big-lama"
    assert lama_tool.margin == 256


def test_empty_mask_fast_path_returns_a_copy(lama_tool, synthetic_batch) -> None:
    """No mask -> no network call: the batch returns as an identical copy.

    An empty mask is skipped before any tensor work, so an un-erased batch is
    never rewritten by the network.
    """
    masks = np.zeros(synthetic_batch.data.shape[:3], dtype=bool)
    out = lama_tool.inpaint(synthetic_batch.data, masks)
    assert out is not synthetic_batch.data
    assert np.array_equal(out, synthetic_batch.data)


def test_masked_fill_changes_only_masked_pixels_and_stays_in_range(lama_tool, synthetic_batch) -> None:
    """A real fill is uint8 in [0, 255]; masked pixels change, the rest is intact.

    Only the boolean-mask pixels may move: unmasked color pixels and the whole
    alpha plane come back byte-identical, and the rewritten pixels stay inside
    the selection's value range.
    """
    masks = _region_masks(synthetic_batch)
    out = lama_tool.inpaint(synthetic_batch.data, masks)
    assert out.shape == synthetic_batch.data.shape
    assert out.dtype == np.uint8
    assert (out[masks] != synthetic_batch.data[masks]).any()
    assert np.array_equal(out[~masks], synthetic_batch.data[~masks])
    assert np.array_equal(out[..., 3], synthetic_batch.data[..., 3])
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 255.0


def test_two_runs_are_byte_identical(lama_tool, synthetic_batch) -> None:
    """Real-torch determinism: the same batch and mask yield the same bytes.

    Whatever the network's internals, calling the adapter twice under
    ``inference_mode`` must reproduce the stored output exactly — the property
    the opt-in generative run and the sweep invariance rely on.
    """
    masks = _region_masks(synthetic_batch)
    first = lama_tool.inpaint(synthetic_batch.data, masks)
    second = lama_tool.inpaint(synthetic_batch.data, masks)
    assert np.array_equal(first, second)


def test_crop_and_pad_to_multiples_of_8(lama_tool) -> None:
    """A 36x36 frame pads both axes to multiples of 8 before the network call.

    The 32x32 synthetic frames are already 8-aligned, so the crop-pad branch
    that rounds a non-multiple-of-8 window up never fires there. Here the
    margin clamps the crop to the whole 36x36 frame, forcing a pad of 4 on
    each axis: the image and mask tensors handed to the network must still
    agree on shape (regression: the image pad was applied to the wrong axis,
    so TorchScript saw a 40-wide image against a 36-wide mask).
    """
    shape = (2, 36, 36, 4)
    frames = np.zeros(shape, dtype=np.uint8)
    frames[0, ..., 0] = 120
    frames[1, ..., 1] = 200
    masks = np.zeros(shape[:3], dtype=bool)
    top, left, height, width = REGION.values()
    masks[:, top : top + height, left : left + width] = True
    out = lama_tool.inpaint(frames, masks)
    assert out.shape == shape
    assert out.dtype == np.uint8
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 255.0


def test_built_from_bare_model_ref_carries_default_margin(cached_big_lama) -> None:
    """``cli._build_models`` on a bare ``ModelRefConfig`` builds the real tool.

    A ``models:`` reference with only ``type`` (no ``params``) instantiates the
    production ``LamaTool`` through the real registry — weights dropped when
    unset, default ``margin`` preserved, checkpoint name recorded.
    """
    validated = JobConfig.model_validate(
        build_config(
            loaders=[build_loader(parquet_path="missing.parquet")],
            output_path="ledger.parquet",
            samples_dir="samples",
            models={"lama": {"type": "lama_inpaint"}},
        )
    )
    tool = cli._build_models(validated)["lama"]
    assert isinstance(tool, LamaTool)
    assert tool.margin == 256
    assert tool.name == "big-lama"
    assert tool.weights == cached_big_lama
