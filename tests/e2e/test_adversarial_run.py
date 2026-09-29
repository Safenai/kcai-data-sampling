"""The opt-in adversarial run: ``fgsm`` end to end against the stub target.

Like the other ``-fgsm`` test modules, this file is skipped (visible) in the
default env and runs against the *real* registry in ``nox -s test-fgsm``: the
stub target is seeded through ``register_model`` (the notebook path), then
``cli.run`` resolves the ``fgsm`` plugin and the assertions check the full
write path — the exact 13-column ledger row (``target_model=stub``,
``family=adversarial``, ``arity=unary``, ``reversible=false``, ``seed=null``,
``params`` exactly ``{epsilon}``), content-addressed uint8 payload files, a
byte-level delta of exactly ``round(ε·255)`` on the stub's sign pixels with
clipping at the extremes, and byte-identical determinism across two runs.
"""

import hashlib
import json
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

pytest.importorskip("kcai_data_sampling_fgsm.api.transformations.fgsm")

pytestmark = pytest.mark.fgsm


def _adversarial_config(root, raw_bytes_data, *, epsilon: float = 2 / 255) -> dict:
    """A job config: one ``fgsm`` against the stub target over the raw sidecar.

    Args:
        root: Output root for the run.
        raw_bytes_data: The raw-bytes parquet fixture path.
        epsilon: The adversarial budget (in ``[0, 1]`` pixel units).

    Returns:
        The config dict, with ``models: stub → stub`` and the transformation
        naming it as its target model.
    """
    return build_config(
        loaders=[build_loader(parquet_path=raw_bytes_data)],
        output_root=root,
        transformations=[{"type": "fgsm", "target_model": "stub", "epsilon": epsilon}],
        models={"stub": {"type": "stub"}},
    )


def _stub_expected(frame: np.ndarray, epsilon: float) -> np.ndarray:
    """Recompute the stub-adversarial output for one source frame, in numpy.

    Mirrors the stub's fixed sign direction (+1 on the red plane wherever the
    green-blue mean is below ``0.5``, 0 elsewhere) and the ``fgsm_step`` math:
    normalize the integer batch by its dtype max, add ``epsilon * sign``, clip
    back to ``[0, 1]``, scale/round/cast. Used to pin the payload bytes without
    calling back into the algorithm under test.

    Args:
        frame: One ``(32, 32, 4)`` uint8 source frame.
        epsilon: The adversarial budget used by the run.

    Returns:
        The expected perturbed frame, ``(32, 32, 4)`` uint8.
    """
    info = np.iinfo(frame.dtype)
    normalized = frame.astype(np.float64) / info.max
    sign = np.zeros_like(normalized)
    dark = normalized[..., 1:3].mean(axis=-1) < 0.5
    sign[..., 0][dark] = 1.0
    stepped = np.clip(normalized + epsilon * np.sign(sign), 0.0, 1.0)
    return np.rint(stepped * info.max).astype(frame.dtype)


def test_adversarial_run_ledger_and_payloads(raw_bytes_data, test_seed, registry_snapshot, tmp_path) -> None:
    """One real ``fgsm`` run records the expected ledger and payloads.

    The summary counts 8 rows x one fgsm = 8; the ledger has exactly the 13
    generator columns, every row is a deterministic unary adversarial step of
    magnitude ``epsilon`` in the ``sign`` direction of the stub target's
    gradient, with a null seed and exactly ``{epsilon}`` in ``params``. Each
    content-hashed PNG is the source frame plus exactly ``round(ε·255)`` on
    the stub's sign pixels, clipped at the extremes.
    """
    register_model("stub", StubTarget)
    epsilon = 2 / 255
    root = tmp_path / "outputs"
    summary = run(_adversarial_config(root, raw_bytes_data, epsilon=epsilon))
    assert summary == {"synthetic": 8}

    table = pq.read_table(str(root / "ledger" / "synthetic.parquet"))
    assert table.column_names == list(GENERATOR_COLUMNS)
    assert table.num_rows == 8
    assert table.column("algorithm").to_pylist() == ["fgsm"] * 8
    assert table.column("family").to_pylist() == ["adversarial"] * 8
    assert table.column("arity").to_pylist() == ["unary"] * 8
    assert table.column("reversible").to_pylist() == [False] * 8
    assert table.column("tool_model").to_pylist() == [None] * 8
    assert table.column("target_model").to_pylist() == ["stub"] * 8
    assert table.column("seed").to_pylist() == [None] * 8
    assert all(json.loads(p) == {"epsilon": epsilon} for p in table.column("params").to_pylist())

    ids = table.column("id").to_pylist()
    assert len(set(ids)) == 8
    assert all(re.fullmatch(r"[0-9a-f]{12}", i) is not None for i in ids)
    parents = table.column("parent_id").to_pylist()
    assert set(parents) == {f"syn_{i:04d}" for i in range(8)}
    parent_by_id = dict(zip(ids, parents, strict=True))

    artifacts = dict(zip(ids, table.column("artifact").to_pylist(), strict=True))
    frames = make_frames(test_seed)

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
        expected = _stub_expected(source, epsilon)
        assert pixels.tobytes() == expected.tobytes()

        # The step moved red sign pixels by exactly round(ε·255), or clipped.
        delta = pixels.astype(np.int16) - source.astype(np.int16)
        move = int(np.rint(epsilon * 255))
        assert move == 2
        dark = (source.astype(np.float64) / 255)[..., 1:3].mean(axis=-1) < 0.5
        assert np.array_equal(delta[..., 0][dark], np.minimum(move, 255 - source[..., 0][dark]))
        assert match.group("c6") == hashlib.sha1(pixels.tobytes()).hexdigest()[:6]

        # Everywhere the sign is 0 — non-red planes, bright pixels, alpha —
        # is the source, byte for byte.
        outside = np.ones((32, 32), dtype=bool)
        outside[dark] = False
        assert np.array_equal(pixels[..., 0][outside], source[..., 0][outside])
        assert np.array_equal(pixels[..., 1:], source[..., 1:])


def test_adversarial_run_is_deterministic(raw_bytes_data, registry_snapshot, tmp_path) -> None:
    """Two runs of the real adversarial job replay byte-for-byte.

    Same selection, same stub target, same budget: the ledger parquet bytes and
    every payload file (name and contents) must be identical.
    """
    register_model("stub", StubTarget)
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert run(_adversarial_config(first, raw_bytes_data)) == {"synthetic": 8}
    assert run(_adversarial_config(second, raw_bytes_data)) == {"synthetic": 8}

    first_ledger = first / "ledger" / "synthetic.parquet"
    second_ledger = second / "ledger" / "synthetic.parquet"
    assert first_ledger.read_bytes() == second_ledger.read_bytes()

    first_payloads = {p.name: p.read_bytes() for p in (first / "synthetic").glob("*.png")}
    second_payloads = {p.name: p.read_bytes() for p in (second / "synthetic").glob("*.png")}
    assert first_payloads == second_payloads
