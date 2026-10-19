"""The opt-in ART run: all six ``art_*`` attacks end to end against the stub target.

Like the other ``-art`` test modules, this file is skipped (visible) in the
default env and runs against the *real* registry in ``nox -s test_art``: the
stub target (``grad`` for the gradient attacks, ``predict`` for the black-box
ones) is seeded through ``register_model``, then ``cli.run`` resolves the six
plugins and the assertions check the full write path: the 13-column ledger
rows (``seed`` null for the gradient attacks and recorded for the black-box
ones, ``params`` exactly the resolved parameters), content-addressed uint8
payloads, the L-infinity budget and the untouched alpha plane, ``art_fgsm``
byte-equal to the local FGSM step, and byte-identical replays across runs and
batch-split knobs.
"""

import collections
import hashlib
import itertools
import json
from pathlib import Path
import re

from kcai_data_sampling_core.utils.registry import register_model
from kcai_data_sampling_job.cli import run
from kcai_data_sampling_job.job import GENERATOR_COLUMNS
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import pytest

from tests.e2e.test_job_run import ARTIFACT_RE
from tests.fixtures.data import make_frames
from tests.fixtures.registries import StubTarget
from tests.utils.configs import build_config, build_loader

pytest.importorskip("kcai_data_sampling_art")

pytestmark = pytest.mark.art

EPSILON = 2 / 255
SEED = 0

#: The six attacks, each with the exact ``params`` its ledger rows record.
ATTACKS = {
    "art_fgsm": {"epsilon": EPSILON},
    "art_pgd": {"epsilon": EPSILON, "max_iter": 10, "norm": "inf", "step_fraction": 0.25},
    "art_mim": {"decay": 1.0, "epsilon": EPSILON, "max_iter": 10, "norm": "inf", "step_fraction": 0.25},
    "art_square": {"epsilon": EPSILON, "max_iter": 100, "p_init": 0.05},
    "art_simba": {"epsilon": EPSILON, "max_iter": 1000},
    "art_hopskipjump": {"init_eval": 10, "init_size": 100, "max_eval": 100, "max_iter": 10, "norm": 2},
}
GRADIENT = {"art_fgsm", "art_pgd", "art_mim"}
BLACK_BOX = {"art_square", "art_simba", "art_hopskipjump"}
#: HopSkipJump has no budget; every other attack stays within ``EPSILON``.
BUDGETED = set(ATTACKS) - {"art_hopskipjump"}

#: The batch-knob matrix of the other invariance tests, and its baseline.
LOAD_SIZES = (4, 8)
TRANSFORM_SIZES = (4, 8)
FLUSH_SIZES = (1, 100)
BASELINE = (8, 8, 100)


def _art_config(
    root: Path,
    raw_bytes_data,
    *,
    load: int = 8,
    transform: int | None = None,
    flush: int = 100,
) -> dict:
    """A job config: the six ART attacks against the stub target over the raw sidecar.

    Args:
        root: Output root for the run.
        raw_bytes_data: The raw-bytes parquet fixture path.
        load: ``load_batch_size`` (loader chunk size).
        transform: ``transform_batch_size`` (runner chunk size).
        flush: ``flush_batch_size`` (writer buffer bound).

    Returns:
        The config dict, with ``models: stub → stub``, the gradient attacks at
        ``EPSILON`` and the black-box ones seeded with ``SEED``, all other
        parameters at their defaults.
    """
    transformations = []
    for algorithm in ATTACKS:
        entry: dict = {"type": algorithm, "target_model": "stub"}
        if algorithm in BUDGETED:
            entry["epsilon"] = EPSILON
        if algorithm in BLACK_BOX:
            entry["seed"] = SEED
        transformations.append(entry)
    return build_config(
        loaders=[build_loader(parquet_path=raw_bytes_data, load_batch_size=load)],
        output_root=root,
        transformations=transformations,
        models={"stub": {"type": "stub"}},
        transform_batch_size=transform,
        flush_batch_size=flush,
    )


def _fgsm_expected(frame: np.ndarray) -> np.ndarray:
    """Recompute the local FGSM step against the stub for one source frame, in numpy.

    Mirrors the stub's fixed sign direction (+1 on the red plane wherever the
    green-blue mean is below ``0.5``, 0 elsewhere): normalize, add
    ``EPSILON * sign``, clip to ``[0, 1]``, scale/round/cast.

    Args:
        frame: One ``(32, 32, 4)`` uint8 source frame.

    Returns:
        The expected attacked frame, ``(32, 32, 4)`` uint8.
    """
    normalized = frame.astype(np.float64) / 255
    sign = np.zeros_like(normalized)
    sign[..., 0][normalized[..., 1:3].mean(axis=-1) < 0.5] = 1.0
    return np.rint(np.clip(normalized + EPSILON * sign, 0.0, 1.0) * 255).astype(np.uint8)


def _outputs(root: Path) -> tuple[bytes, dict[str, bytes]]:
    """The ledger bytes and the payload files (name → bytes) of one run.

    Args:
        root: Output root of the run.

    Returns:
        The ledger parquet bytes and every payload PNG, keyed by file name.
    """
    ledger = (root / "ledger" / "synthetic.parquet").read_bytes()
    return ledger, {p.name: p.read_bytes() for p in (root / "synthetic").glob("*.png")}


def _check_row(row: dict) -> None:
    """Assert one ledger row's facts: a unary adversarial step against ``stub``.

    Args:
        row: One ledger row, as a dict.
    """
    assert row["family"] == "adversarial"
    assert row["arity"] == "unary"
    assert row["reversible"] is False
    assert row["tool_model"] is None
    assert row["target_model"] == "stub"
    assert row["seed"] == (SEED if row["algorithm"] in BLACK_BOX else None)
    assert json.loads(row["params"]) == ATTACKS[row["algorithm"]]
    assert re.fullmatch(r"[0-9a-f]{12}", row["id"]) is not None


def _check_payload(png: Path, by_id: dict[str, dict], frames: list[np.ndarray]) -> None:
    """Assert one payload file: named after its row, alpha kept, within budget.

    Args:
        png: The payload file.
        by_id: The ledger rows, keyed by ``id``.
        frames: The source frames, indexed by the parent id's number.
    """
    match = ARTIFACT_RE.fullmatch(png.name)
    assert match is not None
    row = by_id[match.group("id")]
    assert row["artifact"] == png.name
    pixels = np.asarray(Image.open(png))
    assert pixels.dtype == np.uint8
    assert pixels.shape == (32, 32, 4)
    assert match.group("c6") == hashlib.sha1(pixels.tobytes()).hexdigest()[:6]

    source = frames[int(row["parent_id"].split("_")[1])]
    assert np.array_equal(pixels[..., 3], source[..., 3])
    if row["algorithm"] in BUDGETED:
        assert np.abs(pixels.astype(np.int16) - source.astype(np.int16)).max() <= int(np.rint(EPSILON * 255))
    if row["algorithm"] == "art_fgsm":
        assert pixels.tobytes() == _fgsm_expected(source).tobytes()
    if row["algorithm"] == "art_hopskipjump":
        stub = StubTarget()
        before, after = (stub.predict(x[None] / 255).argmax(axis=1)[0] for x in (source, pixels))
        assert before != after or np.array_equal(pixels, source)


def test_art_run_ledger_and_payloads(raw_bytes_data, test_seed, registry_snapshot, tmp_path) -> None:
    """One run of the six ART attacks records the expected ledger and payloads.

    The summary counts 8 rows x 6 attacks = 48; the ledger has exactly the 13
    generator columns; every row is a unary, non-reversible adversarial step
    against ``stub``, with a null seed for the gradient attacks, ``SEED`` for
    the black-box ones, and exactly the resolved parameters in ``params``.
    Each content-hashed PNG keeps the alpha plane; the budgeted attacks move
    no value by more than ``round(EPSILON * 255)``; ``art_fgsm`` is the local
    FGSM step byte for byte; ``art_hopskipjump`` either flips the stub's top
    class or returns the image unchanged.
    """
    register_model("stub", StubTarget)
    root = tmp_path / "outputs"
    assert run(_art_config(root, raw_bytes_data)) == {"synthetic": 48}

    table = pq.read_table(str(root / "ledger" / "synthetic.parquet"))
    assert table.column_names == list(GENERATOR_COLUMNS)
    assert table.num_rows == 48
    rows = table.to_pylist()
    assert collections.Counter(row["algorithm"] for row in rows) == dict.fromkeys(ATTACKS, 8)
    for row in rows:
        _check_row(row)
    assert len({row["id"] for row in rows}) == 48
    assert {row["parent_id"] for row in rows} == {f"syn_{i:04d}" for i in range(8)}

    by_id = {row["id"]: row for row in rows}
    frames = make_frames(test_seed)
    payloads = sorted((root / "synthetic").glob("*.png"))
    assert len(payloads) == 48
    for png in payloads:
        _check_payload(png, by_id, frames)


def test_art_run_is_deterministic(raw_bytes_data, registry_snapshot, tmp_path) -> None:
    """Two runs of the six ART attacks replay byte for byte, the seeded black-box ones included."""
    register_model("stub", StubTarget)
    first, second = tmp_path / "first", tmp_path / "second"
    assert run(_art_config(first, raw_bytes_data)) == {"synthetic": 48}
    assert run(_art_config(second, raw_bytes_data)) == {"synthetic": 48}
    assert _outputs(first) == _outputs(second)


def test_art_batch_knobs_do_not_change_ledger_or_payloads(raw_bytes_data, registry_snapshot, tmp_path) -> None:
    """Every batch-knob combination replays the ``<8, 8, 100>`` baseline.

    The black-box attacks draw randomness, attacked one image at a time from
    each row's own generator, so which rows share a batch must not change any
    row's output: the full 2x2x2 matrix must leave the ledger and payload
    bytes unchanged.
    """
    register_model("stub", StubTarget)
    load, transform, flush = BASELINE
    baseline_root = tmp_path / "_baseline"
    assert run(_art_config(baseline_root, raw_bytes_data, load=load, transform=transform, flush=flush)) == {
        "synthetic": 48
    }
    baseline = _outputs(baseline_root)

    for load, transform, flush in itertools.product(LOAD_SIZES, TRANSFORM_SIZES, FLUSH_SIZES):
        root = tmp_path / f"l{load}-t{transform}-f{flush}"
        summary = run(_art_config(root, raw_bytes_data, load=load, transform=transform, flush=flush))
        assert summary == {"synthetic": 48}, (load, transform, flush)
        assert _outputs(root) == baseline, (load, transform, flush)
