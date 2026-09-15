"""What the store guarantees, on real outputs of the interface."""

import json
from pathlib import Path

import numpy as np
import pytest

from kcai_data_sampling_core import CropResize, DataSelection, HorizontalFlip, TransformationRunner
from kcai_data_sampling_core.api.unary import UnaryTransformation
from kcai_data_sampling_job import Store


def test_a_selection_is_written_once(store, samples):
    """Same content: idempotent. Different content, same name: refused."""
    same = DataSelection("sel", dataset="comma10k", samples=samples)
    store.write_selection(same)
    store.write_selection(same)

    other = DataSelection("sel", dataset="comma10k", samples=samples[:1])
    with pytest.raises(FileExistsError, match="written once"):
        store.write_selection(other)
    assert len(json.loads(store.selection_path(same).read_text())["samples"]) == len(samples)


def test_storage_adds_its_two_columns_and_nothing_else(store, runner, selection, samples):
    """A row on disk is an Output minus its bitmap, plus where it and its selection landed."""
    outputs = runner.run(HorizontalFlip(), samples)
    rows = store.write(selection, outputs)

    assert len(rows) == len(samples)
    assert set(rows[0]) == set(outputs[0].row()) | {"data_selection_path", "sample_path"}
    assert rows[0]["algorithm"] == "horizontal_flip" and rows[0]["seed"] is None
    assert rows[0]["data_selection_path"] == str(store.selection_path(selection))
    assert Path(rows[0]["sample_path"]).name.startswith(f"{selection.name}__{samples[0].id}__horizontal_flip__")

    written = {p.name for p in store.outputs.iterdir()}
    assert len(written) == len(samples) + 1 and "rows.json" in written  # x′ and the rows, nothing else
    assert store.read(rows[0]).shape == samples[0].x.shape


def test_rows_json_is_a_ledger_not_a_snapshot(store, runner, selection, samples):
    """Two runs in one directory: both sets of rows; a re-run replaces only its own."""
    store.write(selection, runner.run(HorizontalFlip(), samples))
    store.write(selection, runner.run(CropResize({"fraction": 0.4}), samples))
    store.write(selection, runner.run(HorizontalFlip(), samples))  # again

    rows = store.rows()
    assert len(rows) == 2 * len(samples)
    assert sorted({r["algorithm"] for r in rows}) == ["crop_resize", "horizontal_flip"]
    assert len({r["sample_path"] for r in rows}) == len(rows)


def test_two_transformations_of_one_parent_do_not_collide(store, runner, selection, samples):
    """Two settings on one parent: two files, two rows."""
    for fraction in (0.4, 0.5):
        store.write(selection, runner.run(CropResize({"fraction": fraction}), samples))

    rows = store.rows()
    assert len(rows) == 2 * len(samples)
    assert len({r["sample_path"] for r in rows}) == len(rows)
    assert len(list(store.outputs.glob("*.png"))) == 2 * len(samples)


def test_one_identity_two_outputs_keeps_both(store, runner, selection, samples):
    """One identity, two different x′ (non-deterministic kernels): two files,
    two rows identical in every field but `sample_path`."""

    class Unseeded(UnaryTransformation):   # draws without declaring it, stands in for hardware noise
        algorithm = "unseeded"

        def apply(self, xs, rngs):
            return np.clip(xs + np.random.default_rng().normal(0, 0.05, xs.shape), 0, 1)

    for _ in range(2):
        store.write(selection, runner.run(Unseeded(), samples[:1]))

    rows = store.rows()
    assert len(rows) == 2
    fields = [{k: v for k, v in r.items() if k != "sample_path"} for r in rows]
    assert fields[0] == fields[1]
    assert rows[0]["sample_path"] != rows[1]["sample_path"]


def test_a_bit_identical_rerun_is_idempotent(store, runner, selection, samples):
    """Same content, same name: one file, one row."""
    for _ in range(2):
        store.write(selection, runner.run(HorizontalFlip(), samples[:1]))

    assert len(store.rows()) == 1
    assert len(list(store.outputs.glob("*.png"))) == 1


def test_two_selections_do_not_collide(store, samples):
    """The file name carries the selection."""
    for name in ("one", "two"):
        sel = DataSelection(name, dataset="comma10k", samples=samples)
        store.write(sel, TransformationRunner(sel).run(HorizontalFlip()))

    rows = store.rows()
    assert len(rows) == 2 * len(samples)
    assert {Path(r["sample_path"]).name.split("__")[0] for r in rows} == {"one", "two"}


def test_a_row_alone_leads_back_to_x(store, runner, selection, samples):
    """From one row and nothing in memory: the selection by path, the parent
    by id, the reader the selection names, and `x` and its annotation are back."""
    store.write(selection, runner.run(HorizontalFlip(), samples))
    del selection, samples

    row = store.rows()[1]
    parent = store.parent(row)
    x_prime = store.read(row)

    assert np.allclose(x_prime[..., ::-1], parent.x, atol=1 / 255)   # the flip, undone, is the parent
    assert "mask" in parent.y
