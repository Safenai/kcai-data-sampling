"""The opt-in albumentation run: the walkthrough config, driven end to end.

Like the other opt-in e2e modules, this file is deselected in the default env
and runs in ``nox -s test_albumentation``. It loads
``examples/config/walkthrough-albumentation.yaml`` — the living documentation
for the albumentations surface — swaps the comma10k loader for the synthetic
raw-bytes sidecar (the example data is git-ignored) and the outputs for the
test tmp dir, then drives the real ``cli.run`` path.

The automated assertions target the ``random_brightness_contrast`` entry: one
ledger row per input per entry, exact ``params``/``seed`` rows, and payload
bytes reproducible from the ledger facts alone (alpha byte-identical, RGB
replayed from ``(params, seed, row_id)``). The other entry is smoke: the job completes and writes one row per input per
entry — ``blur`` (the second image-only entry) also exercises the alpha-split
payload shape. A second run with different batch knobs replays the first
byte-for-byte — batch invariance for a stochastic transformation.
"""

from collections import Counter
import json
from pathlib import Path

from kcai_data_sampling_job.cli import run
from kcai_data_sampling_job.job import GENERATOR_COLUMNS
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from tests.fixtures.data import make_frames
from tests.utils.configs import build_loader

pytest.importorskip("kcai_data_sampling_albumentation.api.transformations.random_brightness_contrast")

pytestmark = pytest.mark.albumentation

from kcai_data_sampling_albumentation.api.transformations import GaussianBlur, RandomBrightnessContrast  # noqa: E402
from kcai_data_sampling_core.api.selection import Batch  # noqa: E402

#: The living-documentation config this test drives (comma10k swapped out).
WALKTHROUGH = Path(__file__).resolve().parents[2] / "examples" / "config" / "walkthrough-albumentation.yaml"

#: The two entries the walkthrough declares: type name → (seed, row count per
#: input). ``random_brightness_contrast`` sweeps into three entries, so its
#: per-input count is 3.
SEEDS = {"gaussian_blur": 11, "random_brightness_contrast": 42}  # class algorithm attrs (unchanged)
SWEEP_VALUES = [0.1, 0.3, 0.5]


def _walkthrough_config(root: Path, raw_bytes_data: Path, *, load: int | None = None, flush: int | None = None) -> dict:
    """The walkthrough YAML, runnable against the synthetic fixture.

    The example file keeps its comma10k loader (git-ignored, regenerated on
    demand by the notebook); here that loader is swapped for the raw-bytes
    synthetic sidecar and the outputs for ``root``, leaving the transformation
    entries — the documented surface — exactly as published.

    Args:
        root: Output root for this run.
        raw_bytes_data: The raw-bytes parquet fixture path.
        load: Optional ``load_batch_size`` override for the knob replay.
        flush: Optional ``flush_batch_size`` override for the knob replay.

    Returns:
        The config dict, ready for ``cli.run``.
    """
    config = yaml.safe_load(WALKTHROUGH.read_text())
    loader = build_loader(parquet_path=raw_bytes_data)
    if load is not None:
        loader["load_batch_size"] = load
    config["dataloaders"]["loaders"] = [loader]
    outputs = config["operations"]["outputs"]
    outputs["path"] = str(root / "ledger" / "{selection}.parquet")
    outputs["samples_dir"] = str(root / "{selection}")
    if flush is not None:
        outputs["flush_batch_size"] = flush
    return config


def _read_outputs(root: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load the ledger rows and the payload pixels keyed by output id.

    Args:
        root: Output root of a finished run.

    Returns:
        ``(table, payloads)`` — the ledger as numpy columns through
        ``column(...).to_pylist()``, and ``id → (32, 32, 4) uint8 pixels``.
    """
    table = pq.read_table(str(root / "ledger" / "synthetic.parquet"))
    payloads = {}
    for png in sorted((root / "synthetic").glob("*.png")):
        payloads[png.name] = np.asarray(Image.open(png))
    return table, payloads


def _replay(params: dict, frames: np.ndarray, ids: list[str]) -> np.ndarray:
    """Recompute the walkthrough's outputs for one entry from ledger facts alone.

    Args:
        params: The resolved ``params`` recorded in the ledger row.
        frames: The source frames the parent ids point at.
        ids: The parent row ids, in frame order.

    Returns:
        The replayed outputs, ``(B, 32, 32, 4)`` uint8.
    """
    cls = RandomBrightnessContrast if "brightness_limit" in params else GaussianBlur
    transform = cls({"seed": SEEDS[cls.algorithm], **params})
    batch = Batch(name="synthetic", dataset="synthetic", ids=ids, columns=pa.table({"tag": ids}), data=frames)
    return np.stack([output.x for output in transform.transform(batch, value_range=(0, 255))])


def _rows(table) -> list[tuple[str, dict, str]]:
    """The ledger's ``(algorithm, params, parent_id)`` triples, in row order.

    Args:
        table: The ledger table of a finished walkthrough run.

    Returns:
        One tuple per ledger row: the entry name, its parsed ``params`` and
        the input row it descends from.
    """
    return list(
        zip(
            table.column("algorithm").to_pylist(),
            [json.loads(params) for params in table.column("params").to_pylist()],
            table.column("parent_id").to_pylist(),
            strict=True,
        )
    )


@pytest.fixture(scope="module")
def walkthrough(raw_bytes_data, test_seed, tmp_path_factory):
    """One finished walkthrough run, shared by the assertions on its outputs.

    Args:
        raw_bytes_data: The raw-bytes parquet fixture (swapped in for comma10k).
        test_seed: The suite seed the source frames were built from.
        tmp_path_factory: Module-scoped output root for the single run.

    Returns:
        ``(table, payloads, frames)`` — the ledger table, ``artifact name →
        pixels`` and the eight source frames in parent-id order.
    """
    root = tmp_path_factory.mktemp("albumentation_walkthrough")
    assert run(_walkthrough_config(root, raw_bytes_data)) == {"synthetic": 32}
    table, payloads = _read_outputs(root)
    return table, payloads, make_frames(test_seed)


def test_walkthrough_writes_one_row_per_input_per_entry(walkthrough) -> None:
    """The published walkthrough config runs end to end: 8 inputs x 4 entries.

    The config declares one ``blur`` and one ``brightness`` entry swept into
    three values, so the ledger must hold exactly 32 rows under the 13
    generator columns plus the two passthroughs, and every input row must
    descend from each concrete entry exactly once.
    """
    table, payloads, _ = walkthrough

    assert table.column_names == list(GENERATOR_COLUMNS) + ["height", "width"]
    assert table.num_rows == 32
    assert Counter(row[0] for row in _rows(table)) == {
        "gaussian_blur": 8,
        "random_brightness_contrast": 24,
    }
    assert set(table.column("parent_id").to_pylist()) == {f"syn_{i:04d}" for i in range(8)}
    assert len(payloads) == 32


def test_walkthrough_rows_carry_the_procedural_declarations_and_seeds(walkthrough) -> None:
    """Every row records the shared family/arity facts and its entry's seed.

    Every entry is a procedural unary non-reversible transformation with no
    model attached; each carries the seed the walkthrough pinned for it, since
    all albumentations-backed transformations are stochastic — including the
    two Tier-B smoke entries.
    """
    table, _, _ = walkthrough

    assert table.column("family").to_pylist() == ["procedural"] * 32
    assert table.column("arity").to_pylist() == ["unary"] * 32
    assert table.column("reversible").to_pylist() == [False] * 32
    assert table.column("tool_model").to_pylist() == [None] * 32
    assert table.column("target_model").to_pylist() == [None] * 32
    seeds = set(zip(table.column("algorithm").to_pylist(), table.column("seed").to_pylist(), strict=True))
    assert seeds == {
        ("gaussian_blur", 11),
        ("random_brightness_contrast", 42),
    }


def test_walkthrough_ledger_records_the_exact_resolved_params(walkthrough) -> None:
    """The tested entry's rows carry the sweep-expanded scalar and the defaults.

    The sweep expands to three fixed brightness magnitudes applied to all 8
    inputs; ``contrast_limit``, ``ensure_safe_range`` and the always-on ``p``
    default are recorded exactly as the schema resolved them, and the blur
    entry records its pinned pair with the same probability default.
    """
    table, _, _ = walkthrough
    rows = _rows(table)

    tested = [params for algorithm, params, _ in rows if algorithm == "random_brightness_contrast"]
    assert Counter(params["brightness_limit"] for params in tested) == dict.fromkeys(SWEEP_VALUES, 8)
    assert all(params["contrast_limit"] == [-0.2, 0.2] for params in tested)
    assert all(params["ensure_safe_range"] is False for params in tested)
    assert all(params["p"] == 1.0 for params in tested)

    blurred = [params for algorithm, params, _ in rows if algorithm == "gaussian_blur"]
    assert all(params == {"p": 1.0, "sigma_limit": [0.5, 3.0]} for params in blurred)


def test_walkthrough_payloads_keep_alpha_and_replay_from_the_ledger(walkthrough) -> None:
    """Payload bytes are reproducible from the ledger facts alone.

    Every payload is a 4-channel uint8 image of the input's shape, and its
    alpha plane is byte-identical to its source frame. For the two image-only
    entries (``blur``, ``brightness``) that holds
    because the engine splits the alpha off and re-attaches it untouched; the
    two Tier-B entries are dual transforms, which receive the full 4-channel
    row so the alpha plane moves with the pixels — after the shape restore
    (resize/crop back) the result is again input-shaped, with the alpha
    resampled like any other plane. For the tested entry, the recorded
    ``(params, seed, parent_id)`` must recompute the exact payload bytes —
    nothing else is needed to replay the run.
    """
    table, payloads, frames = walkthrough
    ids = table.column("id").to_pylist()
    parents = table.column("parent_id").to_pylist()
    algorithms = table.column("algorithm").to_pylist()
    artifacts = dict(zip(ids, table.column("artifact").to_pylist(), strict=True))
    rows = _rows(table)

    #: The two image-only entries; their alpha plane must survive byte-exact.
    image_only = {"gaussian_blur", "random_brightness_contrast"}

    for output_id, parent, algorithm in zip(ids, parents, algorithms, strict=True):
        payload = payloads[artifacts[output_id]]
        assert payload.shape == (32, 32, 4)
        assert payload.dtype == np.uint8
        if algorithm in image_only:
            assert np.array_equal(payload[..., 3], frames[int(parent.split("_")[1])][..., 3])

    for index, (algorithm, params, parent) in enumerate(rows):
        if algorithm != "random_brightness_contrast":
            continue
        source = frames[int(parent.split("_")[1])]
        assert np.array_equal(payloads[artifacts[ids[index]]], _replay(params, source[None, ...], [parent])[0])


def test_walkthrough_replays_byte_identically_across_batch_knobs(raw_bytes_data, tmp_path) -> None:
    """Different batching, byte-identical outputs — including the stochastic entries.

    The baseline runs at ``<8, 8, 100>``; a second run at ``<4, 4, 1>`` splits
    the same inputs into smaller loader, runner and writer chunks. The summary,
    the ledger parquet bytes and every payload byte must not move: the engine
    draws each row from ``(params, seed, row_id)``, so no chunk boundary can
    leak into the outputs.
    """
    baseline_root = tmp_path / "_baseline"
    assert run(_walkthrough_config(baseline_root, raw_bytes_data, load=8, flush=100)) == {"synthetic": 32}
    baseline_ledger = (baseline_root / "ledger" / "synthetic.parquet").read_bytes()
    baseline_payloads = {p.name: p.read_bytes() for p in (baseline_root / "synthetic").glob("*.png")}

    root = tmp_path / "knobs"
    assert run(_walkthrough_config(root, raw_bytes_data, load=4, flush=1)) == {"synthetic": 32}
    assert (root / "ledger" / "synthetic.parquet").read_bytes() == baseline_ledger
    assert {p.name: p.read_bytes() for p in (root / "synthetic").glob("*.png")} == baseline_payloads
