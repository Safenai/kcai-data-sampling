"""The e2e inventory: one full run per focused question.

Every test drives the real config → ``cli.run`` → ``SamplingJob`` → writers
path in-process and asserts on the ledger parquet and the payload PNGs on
disk. Column checks are exact (the 13 generator columns, in the writer's
order, plus any selected passthrough); pixel checks are byte-level through the
content hash.
"""

import hashlib
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from kcai_data_sampling_job.cli import run
from kcai_data_sampling_job.job import GENERATOR_COLUMNS
from tests.e2e.fixtures.configs import all_forms_config, standard_config, swept_fraction_config

ARTIFACT_RE = re.compile(r"^synthetic__(?P<id>[0-9a-f]{12})__(?P<c6>[0-9a-f]{6})\.png$")


def _read_png(path) -> np.ndarray:
    """Decode a payload PNG back to its ``(H, W, 4)`` uint8 array."""
    return np.asarray(Image.open(path))


def test_full_run_ledger_and_payloads(raw_bytes_data, tmp_path) -> None:
    """One nominal run produces the expected ledger and payload store.

    The summary counts 8 rows × 2 algorithms = 16; the ledger has exactly the
    13 generator columns plus the requested ``height``/``width`` passthrough,
    is metadata-only (no image bytes column), and every row carries the
    deterministic facts — ``seed`` null, sorted-JSON ``params``, 12-hex ``id``.
    The payload store has one content-hashed PNG per row, its ``c6`` derivable
    from the stored pixels, all in the uint8 ``[0, 255]`` space, and each
    ``parent_id`` leads back into the source selection.
    """
    root = tmp_path / "outputs"
    summary = run(standard_config(root, raw_bytes_data, include=["height", "width"]))

    assert summary == {"synthetic": 16}

    table = pq.read_table(str(root / "ledger" / "synthetic.parquet"))
    assert table.column_names == list(GENERATOR_COLUMNS) + ["height", "width"]
    assert table.num_rows == 16
    assert not any(pa.types.is_binary(t) or pa.types.is_large_binary(t) for t in table.schema.types)
    assert table.column("seed").to_pylist() == [None] * 16
    assert all(re.fullmatch(r"[0-9a-f]{12}", i) is not None for i in table.column("id").to_pylist())
    assert len(set(table.column("id").to_pylist())) == 16
    assert table.column("height").to_pylist() == [32] * 16
    assert table.column("width").to_pylist() == [32] * 16

    source_ids = {f"syn_{i:04d}" for i in range(8)}
    assert set(table.column("parent_id").to_pylist()) == source_ids

    payloads = sorted((root / "synthetic").glob("*.png"))
    assert len(payloads) == 16
    artifacts = dict(zip(table.column("id").to_pylist(), table.column("artifact").to_pylist()))
    for png in payloads:
        match = ARTIFACT_RE.fullmatch(png.name)
        assert match is not None
        assert artifacts[match.group("id")] == png.name
        pixels = _read_png(png)
        assert pixels.shape == (32, 32, 4)
        assert pixels.dtype == np.uint8
        assert 0 <= pixels.min() and pixels.max() <= 255
        assert match.group("c6") == hashlib.sha1(pixels.tobytes()).hexdigest()[:6]


def test_include_exclude_passthrough(raw_bytes_data, tmp_path) -> None:
    """``include``/``exclude`` select the exact passthrough column set.

    A ``*`` include pulls the non-generator source columns and ``exclude``
    drops ``source`` again, leaving the 13 generators plus ``height``/``width``
    — and still no pixel/spatial data column.
    """
    root = tmp_path / "outputs"
    run(standard_config(root, raw_bytes_data, include=["*"], exclude=["source"]))

    table = pq.read_table(str(root / "ledger" / "synthetic.parquet"))
    assert table.column_names == list(GENERATOR_COLUMNS) + ["height", "width"]
    assert "source" not in table.column_names


def test_trace_only_no_payload_files(raw_bytes_data, tmp_path) -> None:
    """Trace-only matches the write run's ledger byte-for-byte.

    With ``write_samples: false`` the payload directory never appears, every
    ``artifact`` is still a content hash, and the ledger parquet is
    byte-identical to the full run's — the trace replays the same rows.
    """
    write_root = tmp_path / "write"
    trace_root = tmp_path / "trace"
    file = standard_config(write_root, raw_bytes_data, include=["height", "width"])
    run(file)
    run(standard_config(trace_root, raw_bytes_data, include=["height", "width"], write_samples=False))

    write_ledger = write_root / "ledger" / "synthetic.parquet"
    trace_ledger = trace_root / "ledger" / "synthetic.parquet"
    assert write_ledger.read_bytes() == trace_ledger.read_bytes()

    table = pq.read_table(str(trace_ledger))
    assert table.column_names == list(GENERATOR_COLUMNS) + ["height", "width"]
    artifacts = table.column("artifact").to_pylist()
    assert len(artifacts) == 16
    assert all(ARTIFACT_RE.fullmatch(name) is not None for name in artifacts)
    assert not (trace_root / "synthetic").exists()


def test_sweep_expansion(raw_bytes_data, tmp_path) -> None:
    """A swept ``fraction`` runs as independent algorithms.

    ``{range: [0.2, 0.8], samples: 3, mode: even}`` becomes three crop
    instances, so 8 rows × (flip + 3 crops) = 32 rows, each output carrying
    its resolved fraction in ``params``; the deterministic run replays
    byte-identically into a second output root.
    """
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    summary = run(swept_fraction_config(first_root, raw_bytes_data))
    assert summary == {"synthetic": 32}

    table = pq.read_table(str(first_root / "ledger" / "synthetic.parquet"))
    assert table.num_rows == 32
    params = [__import__("json").loads(v) for v in table.column("params").to_pylist()]
    assert params[:4] == [
        {},
        {"fraction": 0.2, "top": 0, "left": 0},
        {"fraction": 0.5, "top": 0, "left": 0},
        {"fraction": 0.8, "top": 0, "left": 0},
    ]

    run(swept_fraction_config(second_root, raw_bytes_data))
    second_ledger = second_root / "ledger" / "synthetic.parquet"
    assert (first_root / "ledger" / "synthetic.parquet").read_bytes() == second_ledger.read_bytes()


def test_image_column_forms(raw_bytes_data, path_data, tmp_path) -> None:
    """The three image-column forms produce identical rows and pixels.

    The same frames served as raw bytes, relative path + prefix, and absolute
    path yield the same ledger rows (same recipe-derived ids) and the same
    payload bytes, selection by selection.
    """
    root = tmp_path / "outputs"
    summary = run(all_forms_config(root, raw_bytes_data, path_data))
    assert summary == {
        "bytes": 16,
        "relative": 16,
        "absolute": 16,
    }

    def names_by_id(selection: str) -> dict[str, str]:
        """Map output id → artifact name for one selection's ledger."""
        table = pq.read_table(str(root / "ledger" / f"{selection}.parquet"))
        return dict(zip(table.column("id").to_pylist(), table.column("artifact").to_pylist()))

    bytes_by_id = names_by_id("bytes")
    for form in ("relative", "absolute"):
        form_by_id = names_by_id(form)
        # Same recipe → same ids; artifact names differ only by SELECTION prefix.
        assert set(form_by_id.keys()) == set(bytes_by_id.keys())
        for output_id, name in bytes_by_id.items():
            assert form_by_id[output_id].split("__", 1)[1] == name.split("__", 1)[1]
            form_name = form_by_id[output_id]
            assert (root / "bytes" / name).read_bytes() == (root / form / form_name).read_bytes()