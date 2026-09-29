"""The real ``inpaint`` transformation against the real big-lama checkpoint.

Opt-in test module: the whole file is skipped unless ``-lama`` (and torch) is
installed, which is exactly the ``test-lama`` nox env. It lives in in ``unit``
because it pins the transformation's own contract — the region mask math, the
parameter resolution (resolved ``params`` stay exactly the window), the tool
role, and the numeric contract of ``apply`` on real network output — over
whatever the CLI wiring tests already cover.

Batch invariance thought: for ``Inpaint`` the only per-row data is the mask
union, and ``apply`` is one array operation over the whole batch, so a split
would still fill the same crop per union — there is nothing row-order dependent
to vary. The opt-in generative sweep/batch-invariance coverage lives with the
end-to-end tests.
"""

from kcai_data_sampling_core.api.roles import check_model
import numpy as np
import pytest
from tests.fixtures.data import REGION

pytest.importorskip("kcai_data_sampling_lama.api.transformations.inpaint")

pytestmark = pytest.mark.lama

from kcai_data_sampling_lama.api.models.lama import LamaTool  # noqa: E402
from kcai_data_sampling_lama.api.transformations.inpaint import Inpaint  # noqa: E402
from kcai_data_sampling_lama.transformations.inpaint import build_region_mask  # noqa: E402


@pytest.fixture(scope="module")
def lama_tool(cached_big_lama) -> LamaTool:
    """One real ``LamaTool`` shared by all tests of this module.

    Building the tool loads the TorchScript checkpoint — the one allowed suite
    download, fetched once per session by ``cached_big_lama`` and reused.

    Returns:
        A ``LamaTool`` over the cached ``big-lama.pt``.
    """
    return LamaTool(weights=str(cached_big_lama))


def test_region_mask_marks_the_window_only(synthetic_batch) -> None:
    """The mask is exactly the configured window, per row, bool, RGB-plane sized.

    ``build_region_mask`` yields one ``(B, H, W)`` boolean mask: ``True``
    precisely on the ``REGION`` window of every row and ``False`` everywhere
    else — the guarantee the erasure/rewrite assertions lean on.
    """
    masks = build_region_mask(synthetic_batch.data.shape, **REGION)
    assert masks.shape == (synthetic_batch.data.shape[0], 32, 32)
    assert masks.dtype == bool
    top, left, height, width = REGION.values()
    window = (slice(top, top + height), slice(left, left + width))
    assert masks[:, window[0], window[1]].all()
    assert not masks[:, :top].any()
    assert not masks[:, top + height :].any()
    assert not masks[:, :, :left].any()
    assert not masks[:, :, left + width :].any()


def test_region_mask_refuses_out_of_frame_and_degenerate_windows(synthetic_batch) -> None:
    """Windows that leave the frame — or degrade to empty — are refused loudly.

    The frame size is only known at apply time, so the check refuses an
    out-of-frame window instead of silently wrapping; an empty region is refused
    so a mis-typed ``height``/``width`` never erases nothing quietly.
    """
    shape = synthetic_batch.data.shape
    with pytest.raises(ValueError, match="leaves the 32x32 frame"):
        build_region_mask(shape, top=20, left=0, height=16, width=8)
    with pytest.raises(ValueError, match="leaves the 32x32 frame"):
        build_region_mask(shape, top=0, left=28, height=16, width=8)
    with pytest.raises(ValueError, match="top/left >= 0 and height/width > 0"):
        build_region_mask(shape, top=-1, left=0, height=16, width=8)
    with pytest.raises(ValueError, match="top/left >= 0 and height/width > 0"):
        build_region_mask(shape, top=0, left=0, height=0, width=8)
    assert build_region_mask(shape, **REGION).shape == (shape[0], 32, 32)


def test_lama_tool_satisfies_the_tool_protocol(lama_tool) -> None:
    """The real ``LamaTool`` passes the construction-time role checks.

    ``check_model`` (run by every ``Inpaint`` construction) requires a non-empty
    ``name`` and the ``inpaint`` method; `channels = 3` is what keeps the alpha
    plane of an RGBA batch out of the network.
    """
    check_model("inpaint", "tool", lama_tool, ("inpaint",))
    assert isinstance(lama_tool.name, str)
    assert lama_tool.name == "big-lama"
    assert lama_tool.channels == 3


def test_inpaint_resolves_window_params_with_generative_family(lama_tool) -> None:
    """``Inpaint`` resolution keeps exactly the window as ``params``.

    The ``tool_model`` name is consumed before parameter re-validation, so the
    resolved params (and the ledger JSON) carry only the region window. A
    tool's role makes the family generative and the map not reversible; the
    algorithm is deterministic, so the seed records ``None``.
    """
    inp = Inpaint(config={"seed": None, "tool_model": lama_tool, **REGION})
    assert inp.algorithm == "inpaint"
    assert inp.params == dict(REGION)
    assert inp.tool_model is lama_tool
    assert inp.seed is None
    described = inp.describe()
    assert described["family"] == "generative"
    assert described["reversible"] is False
    assert described["tool_model"] == "big-lama"
    assert described["params"] == dict(REGION)


def test_apply_erases_and_rewrites_the_region_leaving_the_rest(synthetic_batch, lama_tool) -> None:
    """The real forward rewrites the window only; everything else is preserved.

    Outside the window the color planes are byte-identical and the whole alpha
    plane is untouched (``channels == 3``) — the "erase a rectangle, fill it
    in, leave the rest" contract. The window itself is rewritten (network
    output, provably in ``[0, 255]`` uint8 by the arity/range checks).
    """
    inp = Inpaint(config={"seed": None, "tool_model": lama_tool, **REGION})
    outputs = inp.transform(synthetic_batch, synthetic_batch.value_range)
    assert len(outputs) == len(synthetic_batch)

    top, left, height, width = REGION.values()
    window = (slice(top, top + height), slice(left, left + width))
    outside = np.ones((32, 32), dtype=bool)
    outside[window] = False
    for out, orig in zip(outputs, synthetic_batch.data, strict=True):
        assert out.x.dtype == np.uint8
        assert out.x.shape == orig.shape
        assert (out.x[..., :3][window] != orig[..., :3][window]).any()
        assert np.array_equal(out.x[..., :3][outside], orig[..., :3][outside])
        assert np.array_equal(out.x[..., 3], orig[..., 3])


def test_apply_stays_in_range_at_both_extremes(lama_tool) -> None:
    """Fully-black and fully-white regions keep the output in the sample space.

    A window that is already at a value-range extreme must not push the batch
    out of ``[0, 255]``: the shape, dtype and range survive, with alpha and
    out-of-window pixels byte-identical (the network sees only the masked
    crop, so it cannot touch the rest).
    """
    frames = np.concatenate(
        [
            np.zeros((2, 32, 32, 4), dtype=np.uint8),
            np.full((2, 32, 32, 4), 255, dtype=np.uint8),
        ]
    )
    top, left, height, width = REGION.values()
    window = (slice(top, top + height), slice(left, left + width))
    inp = Inpaint(config={"seed": None, "tool_model": lama_tool, **REGION})
    out = inp.apply(frames, None)
    assert out.shape == frames.shape
    assert out.dtype == np.uint8
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 255.0
    assert np.array_equal(out[..., 3], frames[..., 3])
    assert np.array_equal(out[..., :3][window], np.clip(out[..., :3][window], 0, 255))


def test_apply_refuses_a_tool_output_of_wrong_shape(synthetic_batch, lama_tool) -> None:
    """A tool returning a broken shape is caught at the first call.

    ``check_output`` fires regardless of the network: the shape must match the
    input batch and the values must be finite, so a regression in the
    adapter's contract fails loudly here instead of corrupting the ledger.
    """
    inp = Inpaint(config={"seed": None, "tool_model": lama_tool, **REGION})
    original = lama_tool.inpaint

    def _broken(xs, masks):
        del masks
        return xs[..., :3]

    lama_tool.inpaint = _broken
    try:
        with pytest.raises(ValueError, match="expected an array of shape"):
            inp.apply(synthetic_batch.data, None)
    finally:
        lama_tool.inpaint = original
