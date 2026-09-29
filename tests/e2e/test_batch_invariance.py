"""The batch-invariance contract: byte-identical jobs across every batch knob.

Determinism as a first-class test, stronger than dqm-ml's
per-metric-tolerance invariance: the surface is deterministic, so the same
selection produces **byte-identical** ledger parquet and **byte-identical**
payload PNG files whatever the combination of ``load_batch_size``,
``transform_batch_size`` and ``flush_batch_size``. Every run is the full
CLI→job→writer path; the baseline is ``<load=8, transform=8, flush=100>``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

from kcai_data_sampling_job.cli import run

from tests.utils.configs import build_config, build_loader

#: The three knobs, dqm-ml-style stress matrix against the 8-row selection.
LOAD_SIZES = (4, 8)
TRANSFORM_SIZES = (4, 8)
FLUSH_SIZES = (1, 100)
BASELINE = (8, 8, 100)


def _run_combo(raw_bytes_data, root: Path, *, load: int, transform: int, flush: int):
    """Run one full job over the synthetic selection with the given knobs.

    Args:
        raw_bytes_data: The raw-bytes parquet fixture path.
        root: Output root for this run.
        load: ``load_batch_size`` (loader chunk size).
        transform: ``transform_batch_size`` (runner chunk size).
        flush: ``flush_batch_size`` (writer buffer bound).

    Returns:
        The ``run(...)`` summary.
    """
    return run(
        build_config(
            loaders=[build_loader(parquet_path=raw_bytes_data, load_batch_size=load)],
            output_root=root,
            transform_batch_size=transform,
            flush_batch_size=flush,
        )
    )


def _ledger_bytes(root: Path) -> bytes:
    """The raw ledger parquet bytes of a selection's run."""
    return (root / "ledger" / "synthetic.parquet").read_bytes()


def _payloads_by_name(root: Path) -> dict[str, bytes]:
    """Map payload artifact name → raw bytes for one run's payload store."""
    return {p.name: p.read_bytes() for p in sorted((root / "synthetic").glob("*.png"))}


def test_batch_knobs_do_not_change_the_ledger_or_payloads(raw_bytes_data, tmp_path) -> None:
    """Every knob combination replays the baseline output byte-for-byte.

    The full 2x2x2 matrix is compared against the ``<8, 8, 100>`` baseline:
    the summary, the exact ledger parquet bytes, and the exact PNG file bytes
    (name + contents) must not move.
    """
    baseline_root = tmp_path / "_baseline"
    baseline_summary = _run_combo(
        raw_bytes_data, baseline_root, **dict(zip(("load", "transform", "flush"), BASELINE, strict=True))
    )
    assert baseline_summary == {"synthetic": 16}
    baseline_ledger = _ledger_bytes(baseline_root)
    baseline_payloads = _payloads_by_name(baseline_root)

    for load, transform, flush in itertools.product(LOAD_SIZES, TRANSFORM_SIZES, FLUSH_SIZES):
        root = tmp_path / f"l{load}-t{transform}-f{flush}"
        summary = _run_combo(raw_bytes_data, root, load=load, transform=transform, flush=flush)
        assert summary == {"synthetic": 16}
        assert _ledger_bytes(root) == baseline_ledger
        assert _payloads_by_name(root) == baseline_payloads
