"""The single tested albumentations method: ``random_brightness_contrast``.

Every integrated method is a shell over one shared engine, so per-method tests
would mostly re-assert upstream behaviour on a frozen dependency. This file
exercises the machinery once, on the transform that trips every behaviour with
crisp assertions: the alpha-split rule on RGBA, per-row seeding (same seed ⇒
byte-identical, batch-order invariant), the generated schema (unknown and
out-of-bounds parameters refused, sweeps expanding through ``JobConfig``,
``p`` defaulting to the sampling always-on ``1.0``), and the unary range
contract on 4-channel and 3-channel batches.
"""

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("kcai_data_sampling_albumentation")

pytestmark = pytest.mark.albumentation

from kcai_data_sampling_albumentation.api.transformations import RandomBrightnessContrast  # noqa: E402
from kcai_data_sampling_albumentation.configs import RandomBrightnessContrastTransformationConfig  # noqa: E402
from kcai_data_sampling_core.api.selection import Batch  # noqa: E402
from kcai_data_sampling_core.models.config import JobConfig  # noqa: E402
from tests.utils.configs import build_config, build_loader, build_sweep  # noqa: E402

#: The payload domain the job hands to every unary transform for images.
VALUE_RANGE = (0, 255)

#: A pair that guarantees a visible change, so the alpha-split assertion cannot
#: flake on a near-zero random draw.
_ALWAYS_CHANGES = {"brightness_limit": [0.5, 0.5], "contrast_limit": [0.0, 0.0]}


def _frames(rows: int = 3, size: int = 8, channels: int = 4, seed: int = 0) -> np.ndarray:
    """Build a deterministic uint8 batch shaped ``(rows, size, size, channels)``.

    Args:
        rows: Number of frames.
        size: Edge length of every frame.
        channels: 4 for RGBA (alpha-split path), 3 for RGB (bypass path).
        seed: Keys the pixel pattern.

    Returns:
        The frames, uint8 in ``[0, 255]``.
    """
    return np.random.default_rng(seed).integers(0, 256, size=(rows, size, size, channels), dtype=np.uint8)


def _batch(frames: np.ndarray, ids: list[str] | None = None) -> Batch:
    """Wrap frames as the decoded batch the runner hands to a transform.

    Args:
        frames: The stacked samples.
        ids: Row identifiers; defaults to ``r0``, ``r1``, ...

    Returns:
        A ``Batch`` carrying one pass-through column and no image-specific axes.
    """
    row_ids = ids if ids is not None else [f"r{i}" for i in range(len(frames))]
    return Batch(
        name="albumentation",
        dataset="unit",
        ids=row_ids,
        columns=pa.table({"tag": row_ids}),
        data=frames,
    )


def _run(transform: RandomBrightnessContrast, batch: Batch) -> np.ndarray:
    """Apply the transform through the unary contract, restacking the outputs.

    Args:
        transform: The configured instance.
        batch: The decoded input batch.

    Returns:
        The outputs, ``(B, H, W, C)`` uint8, in batch order.
    """
    return np.stack([output.x for output in transform.transform(batch, value_range=VALUE_RANGE)])


def test_declarations_are_stochastic_procedural_with_sampling_defaults() -> None:
    """The shell declares the shared contract and the decided sampling default.

    ``family`` stays ``procedural`` (sourced from the shared base, not the
    transform), stochasticity is on, a seed is recorded, and ``p`` defaults to
    ``1.0`` — an entry names an experiment, so always-apply is the sampling
    contract (upstream's ``0.5`` is overridden).
    """
    transform = RandomBrightnessContrast({"seed": 42})

    assert transform.algorithm == "random_brightness_contrast"
    assert transform._family() == "procedural"
    assert transform.arity == "unary"
    assert transform.stochastic is True
    assert transform.reversible is False
    assert transform.seed == 42
    assert transform.params == {
        "p": 1.0,
        "brightness_limit": [-0.2, 0.2],
        "contrast_limit": [-0.2, 0.2],
        "ensure_safe_range": False,
    }


def test_alpha_plane_is_untouched_while_rgb_changes() -> None:
    """RGBA input runs the alpha-split rule: RGB is transformed, alpha bytes stay.

    The plugin never learns about the alpha plane: the engine splits it off,
    runs the image-only transform on RGB and re-attaches the original plane, so
    the output alpha is byte-identical and the payload keeps its shape/dtype.
    """
    frames = _frames()
    transform = RandomBrightnessContrast({"seed": 42, **_ALWAYS_CHANGES})

    out = _run(transform, _batch(frames))

    assert out.shape == frames.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out[..., 3], frames[..., 3])
    assert not np.array_equal(out[..., :3], frames[..., :3])


def test_same_seed_is_byte_identical_and_a_different_seed_draws_differently() -> None:
    """Per-row seeding makes the run replayable from ``(params, seed, row_id)``."""
    frames = _frames()
    batch = _batch(frames)

    first = _run(RandomBrightnessContrast({"seed": 42}), batch)
    replay = _run(RandomBrightnessContrast({"seed": 42}), batch)
    other = _run(RandomBrightnessContrast({"seed": 43}), batch)

    assert np.array_equal(first, replay)
    assert not np.array_equal(first, other)


def test_a_row_is_batch_order_invariant() -> None:
    """A row's output depends on its id, never on the batch it travels in.

    Row ``r1`` transformed alone equals row ``r1`` in a three-row batch, and
    the same holds when the batch order is reversed: the per-row generator is
    keyed by the row id, so sampling is invariant to batching and ordering.
    """
    frames = _frames()
    ids = ["r0", "r1", "r2"]

    full = _run(RandomBrightnessContrast({"seed": 7}), _batch(frames, ids))
    alone = _run(RandomBrightnessContrast({"seed": 7}), _batch(frames[1:2], ["r1"]))
    reversed_out = _run(RandomBrightnessContrast({"seed": 7}), _batch(frames[::-1], ids[::-1]))

    assert np.array_equal(full[1], alone[0])
    assert np.array_equal(full, reversed_out[::-1])


def test_outputs_record_the_replaying_facts_and_stay_in_range() -> None:
    """Every row yields an ``Output`` carrying the seed, params and parent id."""
    batch = _batch(_frames())
    transform = RandomBrightnessContrast({"seed": 42})

    outputs = transform.transform(batch, value_range=VALUE_RANGE)

    assert len(outputs) == len(batch)
    for output, row_id in zip(outputs, batch.ids, strict=True):
        assert output.parent_id == row_id
        assert output.algorithm == "random_brightness_contrast"
        assert output.family == "procedural"
        assert output.seed == 42
        assert output.params == transform.params
        assert int(output.x.min()) >= VALUE_RANGE[0]
        assert int(output.x.max()) <= VALUE_RANGE[1]


def test_three_channel_batches_bypass_the_alpha_split() -> None:
    """An RGB batch keeps working: there is no alpha plane to split off."""
    frames = _frames(channels=3)
    transform = RandomBrightnessContrast({"seed": 42, **_ALWAYS_CHANGES})

    out = _run(transform, _batch(frames))

    assert out.shape == frames.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frames)


def test_a_seed_is_required() -> None:
    """A stochastic transform without a seed is refused at construction."""
    with pytest.raises(ValueError, match=r"stochastic algorithm requires a seed"):
        RandomBrightnessContrast({})


def test_unknown_parameters_are_refused() -> None:
    """The generated schema is the only parameter table: typos fail at load."""
    with pytest.raises(ValueError, match=r"nope"):
        RandomBrightnessContrastTransformationConfig.model_validate({"type": "random_brightness_contrast", "nope": 1})


@pytest.mark.parametrize(
    "params",
    [
        {"brightness_limit": [1.5, 2.0]},
        {"brightness_limit": [0.5, 0.1]},
        {"contrast_limit": [-2.0, 0.0]},
        {"p": 1.5},
    ],
    ids=["above-range", "unordered-pair", "below-range", "probability"],
)
def test_out_of_bounds_parameters_are_refused(params: dict[str, object]) -> None:
    """Pair elements, pair order and ``p`` are all checked at config load."""
    with pytest.raises(ValueError, match=r"must (be|satisfy)"):
        RandomBrightnessContrastTransformationConfig.model_validate({"type": "random_brightness_contrast", **params})


def test_a_scalar_pair_field_revalidates_after_sweep_expansion() -> None:
    """A swept field holds a scalar in the concrete config and still validates.

    ``expand_sweeps`` writes scalar values into the pair field, and the job
    re-validates ``entry.model_dump()`` before constructing the algorithm, so
    the schema must accept a fixed scalar (albumentations broadcasts it to the
    symmetric range).
    """
    config = RandomBrightnessContrastTransformationConfig.model_validate(
        {"type": "random_brightness_contrast", "brightness_limit": 0.3}
    )

    assert config.brightness_limit == 0.3
    assert RandomBrightnessContrastTransformationConfig.model_validate(config.model_dump()).brightness_limit == 0.3


def test_a_brightness_sweep_expands_through_a_job_config(registry_snapshot, tmp_path) -> None:
    """A swept pair field becomes one concrete transformation per drawn scalar."""
    validated = JobConfig.model_validate(
        build_config(
            loaders=[build_loader(parquet_path="missing.parquet")],
            output_path=str(tmp_path / "ledger.parquet"),
            samples_dir=str(tmp_path / "samples"),
            transformations=[
                {"type": "random_brightness_contrast", "seed": 7, "brightness_limit": build_sweep(0.1, 0.3, 3)}
            ],
        )
    )

    resolved = [entry for entry in validated.operations.transformations if entry.type == "random_brightness_contrast"]
    assert [entry.brightness_limit for entry in resolved] == [0.1, 0.2, 0.3]
    assert all(entry.seed == 7 for entry in resolved)
