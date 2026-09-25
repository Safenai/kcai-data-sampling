"""Semantics of the ``horizontal_flip`` transformation.

The flip is the width-axis mirror: it must swap the two-tone bar's halves
byte-exactly, be its own inverse (reversible), preserve the sample space (H, W,
C) and value range, and stay independent of how the batch is split. The class
must come from ``kcai_data_sampling_images.api.transformations`` (the split
surface), with the deterministic declarations.
"""

import numpy as np
import pytest

from kcai_data_sampling_images.api.transformations.horizontal_flip import HorizontalFlip


@pytest.fixture()
def flip() -> HorizontalFlip:
    """A ready ``horizontal_flip`` transformation instance."""
    return HorizontalFlip({})


def test_flip_mirrors_the_width_axis_byte_exactly(synthetic_batch, flip) -> None:
    """A flipped pixel is the mirror of its counterpart, for every pixel.

    Row-for-row comparison against a manual ``[:, :, ::-1, :]`` — no
    resampling, no lossy step may hide behind the class.
    """
    outputs = flip.transform(synthetic_batch, synthetic_batch.value_range)
    assert len(outputs) == len(synthetic_batch)
    for i, output in enumerate(outputs):
        assert np.array_equal(output.x, synthetic_batch.images[i][:, ::-1, :])


def test_flip_swaps_the_bar_halves(synthetic_batch, flip) -> None:
    """The two-tone bar's halves visibly swap: left color appears right.

    Uses the synthetic bar frame (pattern 0): its left edge color must end up
    at the right edge after the flip, and vice versa — the semantic a user
    checks on a rendered plot.
    """
    bar = synthetic_batch.images[0]
    left_color, right_color = bar[:, 0, :], bar[:, -1, :]
    output = flip.transform(synthetic_batch, synthetic_batch.value_range)[0].x
    assert np.array_equal(output[:, -1, :], left_color)
    assert np.array_equal(output[:, 0, :], right_color)


def test_flip_is_its_own_inverse(synthetic_batch, flip) -> None:
    """Flipping twice restores the original bytes: the map is a bijection.

    ``reversible = True`` is a declaration, and this is its operational proof:
    for every row, ``flip(flip(x)) == x`` byte-exact.
    """
    once = flip.transform(synthetic_batch, synthetic_batch.value_range)
    flipped_batch = synthetic_batch.__class__(
        name=synthetic_batch.name,
        dataset=synthetic_batch.dataset,
        ids=synthetic_batch.ids,
        columns=synthetic_batch.columns,
        data=np.stack([o.x for o in once]),
        sample_axes=synthetic_batch.sample_axes,
        value_range=synthetic_batch.value_range,
    )
    twice = flip.transform(flipped_batch, synthetic_batch.value_range)
    for i in range(len(synthetic_batch)):
        assert np.array_equal(twice[i].x, synthetic_batch.images[i])


def test_flip_preserves_shape_space_and_value_range(synthetic_batch, flip) -> None:
    """The output lives in the same space: shape, dtype, and value range.

    The unary contract: ``(B, H, W, C)`` in, ``(B, H, W, C)`` out, uint8, with
    the exact min/max of the input (a mirror cannot change values).
    """
    images = synthetic_batch.images
    outputs = flip.transform(synthetic_batch, synthetic_batch.value_range)
    for i, output in enumerate(outputs):
        assert output.x.shape == images[i].shape
        assert output.x.dtype == np.uint8
        assert output.x.min() == images[i].min()
        assert output.x.max() == images[i].max()


def test_flip_is_independent_of_the_batch_split(synthetic_batch, flip) -> None:
    """Row-by-row flipping gives byte-identical outputs to the whole batch.

    Batch invariance in image form: a split is an execution detail, so
    a row's output never depends on which rows it was batched with.
    """
    whole = flip.transform(synthetic_batch, synthetic_batch.value_range)
    for i in range(len(synthetic_batch)):
        row = synthetic_batch.row(i)
        one = flip.transform(row, row.value_range)[0]
        assert np.array_equal(one.x, whole[i].x)
        assert one.parent_id == whole[i].parent_id


def test_flip_declarations() -> None:
    """The deterministic algorithms declare their nature on the row fields.

    ``algorithm``, ``arity``, ``reversible=True`` (a bijection), family
    derived to ``procedural`` (no model), and ``stochastic=False`` — the four
    declarations a ledger consumer reads.
    """
    declaration = HorizontalFlip({}).describe()
    assert declaration == {
        "algorithm": "horizontal_flip",
        "family": "procedural",
        "arity": "unary",
        "reversible": True,
        "params": {},
        "seed": None,
        "tool_model": None,
        "target_model": None,
    }