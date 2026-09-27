"""Opt-in batch invariance: the adversarial job replays byte-identically.

The same three-knob matrix as the default-env ``test_batch_invariance``, over
the ``fgsm`` pipeline: whatever the ``load_batch_size``,
``transform_batch_size`` and ``flush_batch_size``, the ledger parquet and the
payload PNGs are byte-identical to the ``<8, 8, 100>`` baseline. For ``fgsm``
the batch split cannot leak: ``apply`` is one array operation over the rows it
is given and the stub's gradient depends only on each frame's content, so a
row's output never depends on which other rows it was batched with.
"""

import itertools
from pathlib import Path

import pytest

pytest.importorskip("kcai_data_sampling_fgsm.api.transformations.fgsm")

from kcai_data_sampling_job.cli import run
from kcai_data_sampling_core.utils.registry import register_model
from tests.fixtures.registries import StubTarget
from tests.utils.configs import build_config, build_loader

#: The three knobs, the same stress matrix as the procedural invariance test.
LOAD_SIZES = (4, 8)
TRANSFORM_SIZES = (4, 8)
FLUSH_SIZES = (1, 100)
BASELINE = (8, 8, 100)


def _adversarial_config(
    root: Path,
    raw_bytes_data,
    *,
    load: int,
    transform: int,
    flush: int,
) -> dict:
    """A full adversarial job config over the raw sidecar with the given knobs.

    Args:
        root: Output root for this run.
        raw_bytes_data: The raw-bytes parquet fixture path.
        load: ``load_batch_size`` (loader chunk size).
        transform: ``transform_batch_size`` (runner chunk size).
        flush: ``flush_batch_size`` (writer buffer bound).

    Returns:
        The config dict.
    """
    return build_config(
        loaders=[build_loader(parquet_path=raw_bytes_data, load_batch_size=load)],
        output_root=root,
        transformations=[{"type": "fgsm", "target_model": "stub", "epsilon": 2 / 255}],
        models={"stub": {"type": "stub"}},
        transform_batch_size=transform,
        flush_batch_size=flush,
    )


def test_adversarial_batch_knobs_do_not_change_ledger_or_payloads(
    raw_bytes_data, registry_snapshot, tmp_path
) -> None:
    """Every knob combination replays the baseline adversarial output.

    The full 2×2×2 matrix runs the real ``fgsm`` step; the summary, the exact
    ledger parquet bytes and the exact payload PNG bytes (name + contents) must
    not move from the ``<8, 8, 100>`` baseline.
    """
    register_model("stub", StubTarget)
    baseline_root = tmp_path / "_baseline"
    baseline = dict(zip(("load", "transform", "flush"), BASELINE))
    assert run(_adversarial_config(baseline_root, raw_bytes_data, **baseline)) == {"synthetic": 8}
    baseline_ledger = (baseline_root / "ledger" / "synthetic.parquet").read_bytes()
    baseline_payloads = {p.name: p.read_bytes() for p in (baseline_root / "synthetic").glob("*.png")}

    for load, transform, flush in itertools.product(LOAD_SIZES, TRANSFORM_SIZES, FLUSH_SIZES):
        root = tmp_path / f"l{load}-t{transform}-f{flush}"
        summary = run(
            _adversarial_config(
                root,
                raw_bytes_data,
                load=load,
                transform=transform,
                flush=flush,
            )
        )
        assert summary == {"synthetic": 8}
        assert (root / "ledger" / "synthetic.parquet").read_bytes() == baseline_ledger
        assert {p.name: p.read_bytes() for p in (root / "synthetic").glob("*.png")} == baseline_payloads