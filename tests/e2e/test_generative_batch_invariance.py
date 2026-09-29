"""Opt-in batch invariance: the generative job replays byte-identically.

The same three-knob matrix as the default-env ``test_batch_invariance``, over
the real ``-lama`` pipeline: whatever the ``load_batch_size``,
``transform_batch_size`` and ``flush_batch_size``, the ledger parquet and the
payload PNGs are byte-identical to the ``<8, 8, 100>`` baseline. For
``inpaint`` the batch split cannot leak: ``apply`` is one array operation over
the rows it is given and the mask depends only on the frame shape and the
window, so a row's output never depends on which other rows it was batched
with.
"""

import itertools
from pathlib import Path

from kcai_data_sampling_job.cli import run
import pytest

from tests.fixtures.data import REGION
from tests.utils.configs import build_config, build_loader

pytest.importorskip("kcai_data_sampling_lama.api.transformations.inpaint")

pytestmark = pytest.mark.lama

#: The three knobs, the same stress matrix as the procedural invariance test.
LOAD_SIZES = (4, 8)
TRANSFORM_SIZES = (4, 8)
FLUSH_SIZES = (1, 100)
BASELINE = (8, 8, 100)


def _generative_config(
    root: Path,
    raw_bytes_data,
    *,
    load: int,
    transform: int,
    flush: int,
) -> dict:
    """A full generative job config over the raw sidecar with the given knobs.

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
        transformations=[{"type": "inpaint", "tool_model": "lama", **REGION}],
        models={"lama": {"type": "lama_inpaint"}},
        transform_batch_size=transform,
        flush_batch_size=flush,
    )


def test_generative_batch_knobs_do_not_change_ledger_or_payloads(raw_bytes_data, tmp_path) -> None:
    """Every knob combination replays the baseline generative output.

    The full 2x2x2 matrix runs the real checkpoint; the summary, the exact
    ledger parquet bytes and the exact payload PNG bytes (name + contents) must
    not move from the ``<8, 8, 100>`` baseline.
    """
    baseline_root = tmp_path / "_baseline"
    baseline = dict(zip(("load", "transform", "flush"), BASELINE, strict=True))
    assert run(_generative_config(baseline_root, raw_bytes_data, **baseline)) == {"synthetic": 8}
    baseline_ledger = (baseline_root / "ledger" / "synthetic.parquet").read_bytes()
    baseline_payloads = {p.name: p.read_bytes() for p in (baseline_root / "synthetic").glob("*.png")}

    for load, transform, flush in itertools.product(LOAD_SIZES, TRANSFORM_SIZES, FLUSH_SIZES):
        root = tmp_path / f"l{load}-t{transform}-f{flush}"
        summary = run(
            _generative_config(
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
