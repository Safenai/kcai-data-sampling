"""Each test asserts one claim, on the comma10k sample and YOLOv8n."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from kcai_data_sampling_core import (
    FGSM,
    CropResize,
    CutMix,
    DataSelection,
    HorizontalFlip,
    Inpaint,
    Record,
    Sample,
    TransformationRunner,
)
from kcai_data_sampling_core.api.unary import UnaryTransformation
from kcai_data_sampling_core.utils.io import load_image, save_outputs

EPSILON = 4 / 255
REGION = {"top": 500, "left": 700, "height": 300, "width": 500}


def identity(t):
    """A grouping key, as a downstream module would build one."""
    return (t.algorithm, json.dumps(t.params, sort_keys=True), t.seed)


class Noise(UnaryTransformation):
    """A stochastic algorithm for the tests: one draw per row, from that row's generator."""

    algorithm = "noise"
    stochastic = True

    def apply(self, xs, rngs):
        noise = np.stack([rng.normal(0, self.params["sigma"], xs.shape[1:]) for rng in rngs])
        return xs + noise.astype("float32")


def delta_linf(parent, x_prime):
    return float(np.abs(x_prime - parent.x).max())


# --------------------------------------------------------------- the interface


def test_one_interface_drives_every_family(runner, samples, target, tool):
    """The runner never asks what family it is holding."""
    three = [
        HorizontalFlip(),
        Inpaint({"tool_model": tool, **REGION}),
        FGSM({"target_model": target, "epsilon": EPSILON}),
    ]
    for transformation in three:
        for x_prime, record in runner.run(transformation, samples[:1]):
            assert x_prime.shape == samples[0].x.shape
            assert isinstance(record, Record)

    assert [t.family for t in three] == ["procedural", "generative", "adversarial"]


def test_the_family_is_the_role_of_the_model(target, tool):
    """Derived, never declared: no algorithm can claim a family it does not have."""
    flip = HorizontalFlip()
    fill = Inpaint({"tool_model": tool, **REGION})
    attack = FGSM({"target_model": target, "epsilon": EPSILON})
    assert (flip.tool_model, flip.target_model, flip.family) == (None, None, "procedural")
    assert fill.tool_model.name == "big-lama" and fill.target_model is None and fill.family == "generative"
    assert attack.target_model.name == "yolov8n" and attack.tool_model is None and attack.family == "adversarial"


def test_exactly_the_slot_the_role_names_must_be_filled(target):
    """A target algorithm without a target model, or with a model in the wrong slot, is refused."""
    with pytest.raises(ValueError, match="the role of the model"):
        FGSM({"epsilon": EPSILON})
    with pytest.raises(ValueError, match="the role of the model"):
        FGSM({"tool_model": target, "epsilon": EPSILON})
    with pytest.raises(ValueError, match="the role of the model"):
        HorizontalFlip({"target_model": target})


def test_a_transformation_is_fully_specified(samples):
    """Same algorithm and parameters → same output and same identity."""
    a = CropResize({"fraction": 0.4})
    b = CropResize({"fraction": 0.4})
    c = CropResize({"fraction": 0.5})

    assert np.array_equal(a.transform(samples[0])[0], b.transform(samples[0])[0])
    assert identity(a) == identity(b) != identity(c)


def test_a_deterministic_algorithm_carries_no_seed(samples):
    """`seed = None` is an assertion; a seed offered anyway is refused."""
    _, record = CropResize({"fraction": 0.4}).transform(samples[0])
    assert record.seed is None

    with pytest.raises(ValueError, match="draws no randomness"):
        CropResize({"seed": 7, "fraction": 0.4})


def test_a_stochastic_algorithm_is_identified_by_its_seed(samples):
    """The seed is part of the identity exactly when the algorithm draws."""

    a, b, c = (Noise({"seed": s, "sigma": 0.01}) for s in (7, 7, 8))
    assert np.array_equal(a.transform(samples[0])[0], b.transform(samples[0])[0])
    assert identity(a) == identity(b) != identity(c)
    assert not np.array_equal(a.transform(samples[0])[0], c.transform(samples[0])[0])
    assert a.transform(samples[0])[1].seed == 7


def test_apply_is_one_array_operation_on_the_batch(samples):
    """`apply` takes `(B, *sample)` and returns `(B, *sample)`: no loop over rows."""
    xs = np.stack([s.x for s in samples])
    out = HorizontalFlip().apply(xs, rngs=None)
    assert out.shape == xs.shape and np.array_equal(out[1], samples[1].x[..., ::-1])


def test_the_execution_batch_carries_no_meaning(runner, samples):
    """One batch of three, three of one, or two and one: same rows, same order.

    For a stochastic algorithm too: each output's draws are seeded by the seed
    and its parent's id, not by its place in a batch.
    """
    for transformation in (HorizontalFlip(), CropResize({"fraction": 0.4}), Noise({"seed": 7, "sigma": 0.01})):
        runs = {size: runner.run(transformation, batch_size=size) for size in (None, 1, 2, 3)}
        reference = runs[None]
        for size, results in runs.items():
            assert [r.parent_id for _, r in results] == [r.parent_id for _, r in reference]
            assert all(np.array_equal(a, b) for (a, _), (b, _) in zip(results, reference)), size


def test_an_n_ary_pairs_over_the_selection_not_the_batch(runner, samples):
    """Parents are chosen over the whole selection: `batch_size=1` still finds a partner."""
    runs = {size: runner.run(CutMix(), batch_size=size) for size in (None, 1, 2, 3)}
    reference = runs[None]
    assert len(reference) == len(samples)
    assert [r.parent_id for _, r in reference] == ["00009_f,00012_f", "00012_f,00026_f", "00026_f,00009_f"]
    for size, results in runs.items():
        assert [r.parent_id for _, r in results] == [r.parent_id for _, r in reference], size
        assert all(np.array_equal(a, b) for (a, _), (b, _) in zip(results, reference)), size
    assert reference[0][1].arity == "n-ary"


def test_cutmix_takes_its_right_part_from_the_second_parent(runner, samples):
    x_prime, record = runner.run(CutMix(), samples[:2])[0]
    half = x_prime.shape[-1] // 2
    assert np.array_equal(x_prime[..., :half], samples[0].x[..., :half])
    assert np.array_equal(x_prime[..., half:], samples[1].x[..., half:])
    assert (record.family, record.reversible, record.seed) == ("procedural", False, None)


def test_the_selection_is_what_the_campaign_covers(selection, runner):
    """Running with no explicit samples runs the whole selection."""
    assert len(selection) == 3
    assert len(runner.run(HorizontalFlip())) == len(selection)


def test_the_input_is_never_mutated(samples):
    before = samples[0].x.copy()
    HorizontalFlip().transform(samples[0])
    assert np.array_equal(samples[0].x, before)


# -------------------------------------------- the annotation is out of scope


def test_the_signature_never_changes(samples):
    """`apply` returns x′ alone."""
    x_prime = HorizontalFlip().apply(samples[0].x[None], rngs=None)  # deterministic: no generators
    assert isinstance(x_prime, np.ndarray)


def test_the_annotation_travels_untouched(samples):
    """Not read, not moved, not copied, and nothing about it on the row."""
    sample = samples[0]
    mask_before = sample.y["mask"].copy()

    _, record = HorizontalFlip().transform(sample)

    assert np.array_equal(sample.y["mask"], mask_before)
    assert not [f for f in vars(record) if "annotation" in f or "axes" in f]


# ---------------------------------------------------- no judgement on the row


def test_the_row_carries_no_regime(runner, samples):
    """The regime is a downstream verdict; the row hands the judge `reversible` only."""
    _, record = runner.run(HorizontalFlip(), samples)[0]
    assert not [f for f in vars(record) if "regime" in f or "judge" in f]
    assert record.reversible is True


def test_reversibility_is_a_frozen_declaration(runner, samples):
    """Declared by the algorithm, frozen on the row."""
    _, flip = runner.run(HorizontalFlip(), samples)[0]
    _, zoom = runner.run(CropResize({"fraction": 0.4}), samples)[0]
    assert (flip.reversible, zoom.reversible) == (True, False)
    assert HorizontalFlip.reversible and not CropResize.reversible


def test_the_row_points_at_its_selection(runner, selection, samples):
    """By path, plus the parent's id, no copy of the sample, no transform_id."""
    _, record = runner.run(HorizontalFlip(), samples)[0]
    assert record.data_selection_path == selection.path and Path(record.data_selection_path).exists()
    assert record.parent_id == samples[0].id
    assert not any(f in ("parent_path", "dataset", "transform_id") for f in vars(record))


def test_a_selection_is_written_once(tmp_path, samples):
    """Same content: idempotent. Different content, same name: refused."""
    same = DataSelection("sel", dataset="comma10k", samples=samples)
    same.save(tmp_path / "selections")
    same.save(tmp_path / "selections")

    other = DataSelection("sel", dataset="comma10k", samples=samples[:1])
    with pytest.raises(FileExistsError, match="written once"):
        other.save(tmp_path / "selections")
    assert len(json.loads((tmp_path / "selections" / "sel.json").read_text())["samples"]) == len(samples)


def test_an_unsaved_selection_is_refused(samples):
    """A row references its selection by path, so it must exist on disk first."""
    unsaved = DataSelection("nowhere", dataset="comma10k", samples=samples)
    with pytest.raises(ValueError, match="has not been saved"):
        TransformationRunner(unsaved).run(HorizontalFlip())


# ----------------------------------------------------- the same-space contract


def test_a_unary_transformation_preserves_the_sample_space(samples):
    """Checked on every output: it is what keeps δ defined."""

    class ShrinksTheSample(UnaryTransformation):
        algorithm = "shrinks"

        def apply(self, xs, rngs):
            return xs[..., :400, :400]

    with pytest.raises(ValueError, match="changed the sample space"):
        ShrinksTheSample().transform(samples[0])


def test_crop_resize_keeps_delta_computable(runner, samples):
    """Resampled back, the crop stays comparable to its parent."""
    x_prime, _ = runner.run(CropResize({"fraction": 0.4}), samples)[0]
    assert isinstance(delta_linf(samples[0], x_prime), float)


def test_the_flip_is_its_own_inverse(samples):
    flip = HorizontalFlip()
    once, _ = flip.transform(samples[0])
    twice, _ = flip.transform(Sample("one", once, samples[0].y))
    assert np.allclose(twice, samples[0].x)


# ------------------------------------------------------------- the adversary


def test_fgsm_stays_within_its_budget(runner, samples, target):
    x_prime, _ = runner.run(FGSM({"target_model": target, "epsilon": EPSILON}), samples[:1])[0]
    assert delta_linf(samples[0], x_prime) <= EPSILON + 1e-6


def test_the_gradient_comes_back_at_the_original_resolution(samples, target):
    """The detector runs at 640×640; the gradient comes back at the input's shape."""
    xs = np.stack([s.x for s in samples[:2]])
    g = target.grad(xs)
    assert g.shape == xs.shape
    assert np.isfinite(g).all() and np.count_nonzero(g) > 0
    assert np.allclose(g[0], target.grad(xs[:1])[0], atol=1e-6), "each row's gradient is its own"


def test_magnitude_does_not_decide_the_regime(runner, samples, target):
    """A flip moves every pixel and loses nothing; FGSM moves 4/255."""
    flip = delta_linf(samples[0], runner.run(HorizontalFlip(), samples[:1])[0][0])
    fgsm = delta_linf(samples[0], runner.run(FGSM({"target_model": target, "epsilon": EPSILON}), samples[:1])[0][0])
    assert flip > fgsm


# ------------------------------------------------------------- the tool model


def test_inpaint_touches_only_the_region_and_invents_the_rest(runner, samples, tool):
    """Outside the rectangle, the parent to the pixel; inside, content the model produced."""
    x_prime, record = runner.run(Inpaint({"tool_model": tool, **REGION}), samples[:1])[0]
    t, l, h, w = (REGION[k] for k in ("top", "left", "height", "width"))
    inside = (slice(None), slice(t, t + h), slice(l, l + w))
    outside = np.ones(x_prime.shape[-2:], dtype=bool)
    outside[t : t + h, l : l + w] = False

    assert np.array_equal(x_prime[:, outside], samples[0].x[:, outside])
    assert np.abs(x_prime[inside] - samples[0].x[inside]).mean() > 0.01
    assert (record.family, record.tool_model, record.reversible, record.seed) == ("generative", "big-lama", False, None)


def test_model_backed_transformations_are_batch_invariant(runner, samples, target, tool):
    """The networks see the batch at once; a row's output still does not depend on its neighbours.
    (A small region for LaMa: its memory grows with the batch, which is what `batch_size` is for.)"""
    small = {"top": 500, "left": 700, "height": 100, "width": 100}
    for t in (FGSM({"target_model": target, "epsilon": EPSILON}), Inpaint({"tool_model": tool, **small})):
        together = runner.run(t, samples[:2])
        alone = runner.run(t, samples[:2], batch_size=1)
        assert all(np.allclose(a, b, atol=1e-5) for (a, _), (b, _) in zip(together, alone)), t.algorithm


# ---------------------------------------------------------------------- the row


def test_the_row_carries_no_measurement(runner, samples):
    """Anything recomputable afterwards is not stored."""
    _, record = runner.run(HorizontalFlip(), samples)[0]
    assert not [f for f in vars(record) if f.startswith(("delta", "psnr", "ssim", "clipped"))]


def test_the_row_freezes_what_the_algorithm_declared(runner, samples):
    """A lookup from `algorithm` gives what the class declares now; the row says what ran."""
    _, record = runner.run(HorizontalFlip(), samples)[0]
    assert (record.algorithm, record.family, record.arity, record.reversible) == \
        ("horizontal_flip", "procedural", "unary", True)


def test_run_save_reload(tmp_path, runner, selection, samples):
    rows = save_outputs(tmp_path / "outputs", runner.run(HorizontalFlip(), samples))

    assert len(rows) == len(samples)
    assert rows[0]["algorithm"] == "horizontal_flip"
    assert rows[0]["family"] == "procedural"
    assert rows[0]["tool_model"] is None and rows[0]["target_model"] is None
    assert rows[0]["seed"] is None
    assert rows[0]["data_selection_path"] == selection.path
    assert Path(rows[0]["sample_path"]).name.startswith(f"{selection.name}__{samples[0].id}__horizontal_flip__")

    written = {p.name for p in (tmp_path / "outputs").iterdir()}
    assert len(written) == len(samples) + 1 and "rows.json" in written  # x′ and the rows, nothing else
    assert load_image(rows[0]["sample_path"]).shape == samples[0].x.shape


def test_rows_json_is_a_ledger_not_a_snapshot(tmp_path, runner, selection, samples):
    """Two runs in one directory: both sets of rows; a re-run replaces only its own."""
    save_outputs(tmp_path / "outputs", runner.run(HorizontalFlip(), samples))
    save_outputs(tmp_path / "outputs", runner.run(CropResize({"fraction": 0.4}), samples))
    save_outputs(tmp_path / "outputs", runner.run(HorizontalFlip(), samples))  # again

    rows = json.loads((tmp_path / "outputs" / "rows.json").read_text())
    assert len(rows) == 2 * len(samples)
    assert sorted({r["algorithm"] for r in rows}) == ["crop_resize", "horizontal_flip"]
    assert len({r["sample_path"] for r in rows}) == len(rows)


def test_two_transformations_of_one_parent_do_not_collide(tmp_path, runner, samples):
    """Two settings on one parent: two files, two rows."""
    for fraction in (0.4, 0.5):
        save_outputs(tmp_path / "outputs", runner.run(CropResize({"fraction": fraction}), samples))

    rows = json.loads((tmp_path / "outputs" / "rows.json").read_text())
    assert len(rows) == 2 * len(samples)
    assert len({r["sample_path"] for r in rows}) == len(rows)
    assert len(list((tmp_path / "outputs").glob("*.png"))) == 2 * len(samples)


def test_one_identity_two_outputs_keeps_both(tmp_path, runner, samples):
    """One identity, two different x′ (non-deterministic kernels): two files,
    two rows identical in every field but `sample_path`."""

    class Unseeded(UnaryTransformation):   # draws without declaring it, stands in for hardware noise
        algorithm = "unseeded"

        def apply(self, xs, rngs):
            return np.clip(xs + np.random.default_rng().normal(0, 0.05, xs.shape), 0, 1)

    for _ in range(2):
        save_outputs(tmp_path / "outputs", runner.run(Unseeded(), samples[:1]))

    rows = json.loads((tmp_path / "outputs" / "rows.json").read_text())
    assert len(rows) == 2
    fields = [{k: v for k, v in r.items() if k != "sample_path"} for r in rows]
    assert fields[0] == fields[1]
    assert rows[0]["sample_path"] != rows[1]["sample_path"]


def test_a_bit_identical_rerun_is_idempotent(tmp_path, runner, samples):
    """Same content, same name: one file, one row."""
    for _ in range(2):
        save_outputs(tmp_path / "outputs", runner.run(HorizontalFlip(), samples[:1]))

    rows = json.loads((tmp_path / "outputs" / "rows.json").read_text())
    assert len(rows) == 1
    assert len([p for p in (tmp_path / "outputs").iterdir() if p.suffix == ".png"]) == 1


def test_two_selections_do_not_collide(tmp_path, samples):
    """The file name carries the selection."""
    for name in ("one", "two"):
        sel = DataSelection(name, dataset="comma10k", samples=samples)
        sel.save(tmp_path / "selections")
        save_outputs(tmp_path / "outputs", TransformationRunner(sel).run(HorizontalFlip()))

    rows = json.loads((tmp_path / "outputs" / "rows.json").read_text())
    assert len(rows) == 2 * len(samples)
    assert {Path(r["sample_path"]).name.split("__")[0] for r in rows} == {"one", "two"}


def test_a_row_alone_leads_back_to_x(tmp_path, runner, selection, samples):
    """From one row and nothing in memory: the selection by path, the parent
    by id, the reader the selection names, and `x` and its annotation are back."""
    save_outputs(tmp_path / "outputs", runner.run(HorizontalFlip(), samples))
    del selection, samples

    row = json.loads((tmp_path / "outputs" / "rows.json").read_text())[1]
    written = json.loads(Path(row["data_selection_path"]).read_text())
    entry = next(s for s in written["samples"] if s["id"] == row["parent_id"])
    reader = importlib.import_module(f"kcai_data_sampling_core.datasets.{written['dataset']}")
    parent = reader.load_sample(entry["id"], entry["source"])
    x_prime = load_image(row["sample_path"])

    assert np.allclose(x_prime[..., ::-1], parent.x, atol=1 / 255)   # the flip, undone, is the parent
    assert "mask" in parent.y

