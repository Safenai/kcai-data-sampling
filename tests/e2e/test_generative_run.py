"""The opt-in generative run: real ``-lama`` end to end.

Like the other ``-lama`` test modules, this file is skipped (visible) in the
default env and runs against the *real* registry in ``nox -s test_lama``:
``cli.run`` resolves the ``lama_inpaint`` plugin, the real checkpoint runs over
the synthetic selection, and the assertions check the full write path — the
13-column ledger row (``tool_model=big-lama``, ``family=generative``,
``arity=unary``, ``reversible=false``, ``seed=null``, ``params`` exactly the
window), content-addressed payload files, byte-identity of every unmasked pixel
(and the alpha plane) with the source, and exact determinism across two runs.
"""

import hashlib
import json
import re

from kcai_data_sampling_job.cli import run
from kcai_data_sampling_job.job import GENERATOR_COLUMNS
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import pytest

from tests.e2e.test_job_run import ARTIFACT_RE
from tests.fixtures.data import REGION, make_frames
from tests.utils.configs import build_config, build_loader

pytest.importorskip("kcai_data_sampling_lama.api.transformations.inpaint")

pytestmark = pytest.mark.lama


def _generative_config(root, raw_bytes_data, *, load: int | None = None) -> dict:
    """A job config: one ``inpaint`` on the REGION window over the raw sidecar.

    Args:
        root: Output root for the run.
        raw_bytes_data: The raw-bytes parquet fixture path.
        load: Optional ``load_batch_size`` (unset uses the suite default).

    Returns:
        The config dict, with ``models: lama → lama_inpaint`` and the
        transformation naming it as its tool model.
    """
    return build_config(
        loaders=[build_loader(parquet_path=raw_bytes_data, load_batch_size=load)],
        output_root=root,
        transformations=[{"type": "inpaint", "tool_model": "lama", **REGION}],
        models={"lama": {"type": "lama_inpaint"}},
    )


def test_generative_run_ledger_and_payloads(raw_bytes_data, test_seed, tmp_path) -> None:
    """One real generative run records the expected ledger and payloads.

    The summary counts 8 rows x one inpaint = 8; the ledger has exactly the 13
    generator columns, every row is a deterministic unary generative
    transformation of the ``big-lama`` tool with a null seed and the region
    window alone in ``params``. Each content-hashed PNG is the source frame with
    precisely the masked window rewritten.
    """
    root = tmp_path / "outputs"
    summary = run(_generative_config(root, raw_bytes_data))
    assert summary == {"synthetic": 8}

    table = pq.read_table(str(root / "ledger" / "synthetic.parquet"))
    assert table.column_names == list(GENERATOR_COLUMNS)
    assert table.num_rows == 8
    assert table.column("algorithm").to_pylist() == ["inpaint"] * 8
    assert table.column("family").to_pylist() == ["generative"] * 8
    assert table.column("arity").to_pylist() == ["unary"] * 8
    assert table.column("reversible").to_pylist() == [False] * 8
    assert table.column("tool_model").to_pylist() == ["big-lama"] * 8
    assert table.column("target_model").to_pylist() == [None] * 8
    assert table.column("seed").to_pylist() == [None] * 8
    assert all(json.loads(p) == dict(REGION) for p in table.column("params").to_pylist())

    ids = table.column("id").to_pylist()
    assert len(set(ids)) == 8
    assert all(re.fullmatch(r"[0-9a-f]{12}", i) is not None for i in ids)
    parents = table.column("parent_id").to_pylist()
    assert set(parents) == {f"syn_{i:04d}" for i in range(8)}
    parent_by_id = dict(zip(ids, parents, strict=True))

    artifacts = dict(zip(ids, table.column("artifact").to_pylist(), strict=True))
    frames = make_frames(test_seed)
    top, left, height, width = REGION.values()
    window = (slice(top, top + height), slice(left, left + width))
    outside = np.ones((32, 32), dtype=bool)
    outside[window] = False

    payloads = sorted((root / "synthetic").glob("*.png"))
    assert len(payloads) == 8
    for png in payloads:
        match = ARTIFACT_RE.fullmatch(png.name)
        assert match is not None
        assert artifacts[match.group("id")] == png.name
        pixels = np.asarray(Image.open(png))
        assert pixels.dtype == np.uint8
        assert pixels.shape == (32, 32, 4)
        source = frames[int(parent_by_id[match.group("id")].split("_")[1])]
        # The window is rewritten; everywhere else — and the alpha plane — is the source.
        assert not np.array_equal(pixels[..., :3][window], source[..., :3][window])
        assert np.array_equal(pixels[..., :3][outside], source[..., :3][outside])
        assert np.array_equal(pixels[..., 3], source[..., 3])
        assert match.group("c6") == hashlib.sha1(pixels.tobytes()).hexdigest()[:6]


def test_generative_run_is_deterministic(raw_bytes_data, tmp_path) -> None:
    """Two runs of the real generative job replay byte-for-byte.

    Same selection, same checkpoint, same window: the ledger parquet bytes and
    every payload file (name and contents) must be identical.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert run(_generative_config(first, raw_bytes_data)) == {"synthetic": 8}
    assert run(_generative_config(second, raw_bytes_data)) == {"synthetic": 8}

    first_ledger = first / "ledger" / "synthetic.parquet"
    second_ledger = second / "ledger" / "synthetic.parquet"
    assert first_ledger.read_bytes() == second_ledger.read_bytes()

    first_payloads = {p.name: p.read_bytes() for p in (first / "synthetic").glob("*.png")}
    second_payloads = {p.name: p.read_bytes() for p in (second / "synthetic").glob("*.png")}
    assert first_payloads == second_payloads
