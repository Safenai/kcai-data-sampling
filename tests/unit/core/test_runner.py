"""Runner and Output contract on a generic ``Batch``.

The runner is datatype-agnostic: these tests exercise it with hand-built
generic ``Batch`` objects (``sample_axes``/``value_range`` may be ``None``)
and with small local algorithm classes, so the contract is pinned without
pulling image code in — exactly as core must stay independent of ``-images``.
The seed rules are asserted here at the instance level.

Model-role (tool) coverage lives elsewhere on purpose: the injected-stub
resolution is pinned in ``tests/unit/job/test_model_resolution.py``, and the
real ``Inpaint`` + ``big-lama`` forward in the opt-in
``tests/unit/images/test_inpaint.py`` — so this module stays model-free while
the slot/`check_output` mechanics are still exercised.
"""

import re
from typing import Literal

from kcai_data_sampling_core.api.output import output_identity
from kcai_data_sampling_core.api.selection import Batch
from kcai_data_sampling_core.api.unary import UnaryTransformation
from kcai_data_sampling_core.utils.runner import TransformationRunner
import numpy as np
import pyarrow as pa
import pydantic
from pydantic import BaseModel, ConfigDict
import pytest

ID_RE = re.compile(r"^[0-9a-f]{12}$")


class _TestParams(BaseModel):
    """Parameter schema shared by the local test algorithms (like `Config`)."""

    model_config = ConfigDict(extra="forbid")


class _AddConstConfig(_TestParams):
    """Deterministic test algorithm: add a constant to every sample."""

    amount: float
    mode: Literal["plain", "double"] = "plain"


class _AddConst(UnaryTransformation):
    """Add ``amount`` to every sample: a deterministic unary map."""

    algorithm = "test_add_const"
    Config = _AddConstConfig
    reversible = False
    stochastic = False
    clips = False

    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        del rngs
        return xs + self.params["amount"]


class _ClippingAdd(_AddConst):
    """Same as `_AddConst` but clips to the selection's value range."""

    algorithm = "test_clipping_add"
    clips = True


class _JitterConfig(_TestParams):
    """Stochastic test algorithm config: ``sigma`` scales the draw."""

    sigma: int = 1


class _Jitter(UnaryTransformation):
    """Per-pixel ± noise: deterministic only for a fixed seed + ids."""

    algorithm = "test_jitter"
    Config = _JitterConfig
    reversible = False
    stochastic = True
    clips = True

    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        out = xs.copy()
        for i in range(len(out)):
            delta = rngs[i].integers(-self.params["sigma"], self.params["sigma"] + 1)
            out[i] = out[i] + delta
        return out


def _generic_batch(size: int = 8, value_range: tuple[float, float] | None = None) -> Batch:
    """A generic float batch: no image axes, one id and column per row.

    ``value_range`` defaults to ``None`` (unchecked) so contract tests are not
    constrained by a range; range-enforcement tests pass an explicit one.
    """
    data = np.repeat(np.linspace(0.0, 1.0, size)[:, None], 3, axis=1).astype(np.float64)
    return Batch(
        name="generic",
        dataset="unit",
        ids=[f"g{i:03d}" for i in range(size)],
        columns=pa.table({"tag": [f"tag{i}" for i in range(size)]}),
        data=data,
        sample_axes=None,
        value_range=value_range,
    )


def test_outputs_one_per_row_with_derived_lineage() -> None:
    """Each input row maps to exactly one Output carrying its parent id.

    Applies a deterministic transform over a generic batch and checks the
    row-for-row contract: parent ids echo the batch ids, the output array is
    the transformed sample, and the derived ``id`` is the 12-hex sha1 of the
    row fields.
    """
    batch = _generic_batch()
    outputs = TransformationRunner(batch).run(_AddConst({"amount": 1.0}))

    assert len(outputs) == len(batch)
    for i, output in enumerate(outputs):
        assert output.parent_id == batch.ids[i]
        assert np.array_equal(output.x, batch.data[i] + 1.0)
        row = output.row()
        assert ID_RE.fullmatch(output.id)
        assert ID_RE.fullmatch(row["id"])
        assert output.id == output_identity({k: v for k, v in row.items() if k != "id"})


def test_output_row_carries_exactly_the_declared_fields() -> None:
    """The ledger row is the declarations + replay fields, nothing else.

    ``parent_id``, the frozen declarations (algorithm/family/arity/reversible),
    the replay params/seed and the model slots, plus the derived ``id`` is the
    complete row — no pixel data, no paths, no judgement.
    """
    output = TransformationRunner(_generic_batch()).run(_AddConst({"amount": 0.5}))[0]
    assert set(output.row()) == {
        "id",
        "parent_id",
        "algorithm",
        "family",
        "arity",
        "reversible",
        "params",
        "seed",
        "tool_model",
        "target_model",
    }
    assert output.row()["algorithm"] == "test_add_const"
    assert output.row()["family"] == "procedural"
    assert output.row()["arity"] == "unary"
    assert output.row()["reversible"] is False
    assert output.row()["tool_model"] is None
    assert output.row()["target_model"] is None


def test_output_id_is_deterministic() -> None:
    """Equal rows hash to the same id; nothing environment-dependent leaks in.

    Two independent outputs produced from the same row fields share the id,
    which is what makes ledger dedup and byte-identical runs possible.
    """
    a = TransformationRunner(_generic_batch()).run(_AddConst({"amount": 2.0}))[0]
    b = TransformationRunner(_generic_batch()).run(_AddConst({"amount": 2.0}))[0]
    assert a.id == b.id
    assert a.row() == b.row()


def test_params_resolve_defaults_through_the_registered_schema() -> None:
    """Completing the config fills the schema defaults into ``params``.

    Only the algorithm's own parameters reach the row: the ``mode`` default is
    materialized, while the config keys that don't belong to the algorithm
    (name/type/seed/...) are split off before resolution.
    """
    transformation = _AddConst({"amount": 3.0})
    assert transformation.params == {"amount": 3.0, "mode": "plain"}
    assert "seed" not in transformation.params


def test_unknown_or_missing_params_are_refused() -> None:
    """A config that does not match the algorithm's schema fails at load."""
    with pytest.raises(pydantic.ValidationError):
        _AddConst({"amount": 1.0, "unknown": 0})
    with pytest.raises(pydantic.ValidationError):
        _AddConst({})


def test_value_range_is_enforced_unless_clips_declared() -> None:
    """An out-of-range output is refused; a clipping algorithm clips instead.

    The generic batch declares ``value_range``; an undeclared overflow must
    never reach the ledger, while `_ClippingAdd` (``clips=True``) is allowed
    to leave the range and is clamped back to it.
    """
    transformation = _AddConst({"amount": 1.5})
    with pytest.raises(ValueError, match="left the selection's value range"):
        TransformationRunner(_generic_batch(value_range=(0.0, 1.0))).run(transformation)

    clipped = TransformationRunner(_generic_batch(value_range=(0.0, 1.0))).run(_ClippingAdd({"amount": 1.5}))
    full = TransformationRunner(_generic_batch()).run(_ClippingAdd({"amount": 1.5}))
    assert (np.stack([o.x for o in clipped]) == 1.0).all()
    assert not np.isclose(full[3].x, 1.0).all()


def test_run_on_subset_matches_the_full_batch_run() -> None:
    """Running row-by-row gives byte-identical outputs to the whole batch.

    The batch split is an execution detail only: a
    deterministic transformation produces the same outputs in the same order
    whether the runner sees one batch or those same rows one at a time.
    """
    batch = _generic_batch()
    runner = TransformationRunner(batch)
    transformation = _AddConst({"amount": 1.0})

    whole = runner.run(transformation)
    one_at_a_time = [runner.run(transformation, batch.row(i))[0] for i in range(len(batch))]

    assert [o.parent_id for o in whole] == [o.parent_id for o in one_at_a_time]
    assert all(np.array_equal(whole[i].x, one_at_a_time[i].x) for i in range(len(batch)))


def test_stochastic_run_is_seed_and_id_reproducible() -> None:
    """A stochastic algorithm draws per row, reproducibly from seed + ids.

    Same seed over the same selection must replay the exact same outputs; the
    draw is keyed on the parent id and the resolved params, so a different
    seed (or, below, different ids) diverges.
    """
    batch = _generic_batch()
    first = TransformationRunner(batch).run(_Jitter({"seed": 7}))
    second = TransformationRunner(batch).run(_Jitter({"seed": 7}))
    assert [np.array_equal(a.x, b.x) for a, b in zip(first, second, strict=True)] == [True] * len(batch)
    assert [a.id for a in first] == [b.id for b in second]

    diverged = TransformationRunner(batch).run(_Jitter({"seed": 9}))
    assert any(not np.array_equal(a.x, b.x) for a, b in zip(first, diverged, strict=True))


def test_stochastic_subset_run_matches_full_run() -> None:
    """Row-order independence holds for stochastic algorithms too.

    Each row draws only from its own (seed, id, params) key, so splitting the
    batch never changes what a row outputs — the property a future parallel
    runner inherits.
    """
    batch = _generic_batch()
    runner = TransformationRunner(batch)
    transformation = _Jitter({"seed": 11})

    whole = runner.run(transformation)
    row_runs = [runner.run(transformation, batch.row(i))[0] for i in range(len(batch))]
    assert all(np.array_equal(whole[i].x, row_runs[i].x) for i in range(len(batch)))


def test_relaxed_seed_rule_deterministic_records_none() -> None:
    """A deterministic algorithm records None whether or not a seed was given.

    An explicit seed on a deterministic algorithm is accepted and ignored
    — the row says ``seed: None``, because there is no draw to replay.
    """
    without = _AddConst({"amount": 1.0})
    with_seed = _AddConst({"seed": 5, "amount": 1.0})
    assert without.seed is None
    assert with_seed.seed is None
    assert with_seed.describe()["seed"] is None


def test_relaxed_seed_rule_stochastic_records_its_seed() -> None:
    """A stochastic algorithm records the seed it drew from."""
    transformation = _Jitter({"seed": 7})
    assert transformation.seed == 7
    assert transformation.describe()["seed"] == 7


def test_stochastic_without_seed_is_refused_at_construction() -> None:
    """A stochastic algorithm with no seed cannot be built.

    Refusing at construction (rather than silently drawing 0) is what keeps
    every ledger seed column meaningful: a non-null seed always names the
    draw that replayed the row.
    """
    with pytest.raises(ValueError, match="requires a seed"):
        _Jitter({})
