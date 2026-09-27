"""The clip-dtype pin: integer outputs survive a float value range.

The parquet loader's selection declares a **float** ``(0.0, 255.0)`` value
range, and ``np.clip`` upcasts a uint8 array to float64 when the clip bounds
are Python floats. The ``fgsm`` algorithm is the first delivered one that
declares ``clips``, so its outputs reached the payload writer as float64 and
were refused ("image arrays must be uint8"). The base ``fit_to_range`` now
preserves an integer input's dtype through the clip; this module pins that
contract directly and through the full CLI→runner→writer path.
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry
from kcai_data_sampling_job.cli import run
from tests.fixtures.registries import ClipsIdentityTransformation
from tests.utils.configs import build_config, build_loader


def test_clip_preserves_integer_dtype_under_float_range() -> None:
    """A ``clips`` algorithm keeps a uint8 output uint8 through float bounds.

    The value range the parquet loader declares is floats ``(0.0, 255.0)``;
    the pre-fix clip upcast the uint8 output to float64 and every downstream
    encoder refused it. The dtype is preserved, and the clipped values stay
    inside the range — including where the clip actually cuts the batch.
    """
    transform = ClipsIdentityTransformation()
    out = np.arange(256, dtype=np.uint8).reshape(4, 8, 8)

    kept = transform.fit_to_range(out, (0.0, 255.0))
    assert kept.dtype == np.uint8
    np.testing.assert_array_equal(kept, out)

    clipped = transform.fit_to_range(out, (16.0, 239.0))
    assert clipped.dtype == np.uint8
    assert clipped.min() == 16
    assert clipped.max() == 239
    np.testing.assert_array_equal(clipped, np.clip(out, 16, 239))


def test_clip_float_control_stays_floating() -> None:
    """A float output stays floating through the same clip path.

    Dtype preservation is scoped to integer inputs: float outputs keep the
    clip's floating result (the phase-5 bug concerns the uint8 pipeline only).
    """
    transform = ClipsIdentityTransformation()
    out = np.arange(0.0, 288.0, 16.0, dtype=np.float32).reshape(3, 6)

    clipped = transform.fit_to_range(out, (0.0, 255.0))
    assert np.issubdtype(clipped.dtype, np.floating)
    assert clipped.min() >= 0.0
    assert clipped.max() <= 255.0


def test_clips_job_writes_uint8_payloads(raw_bytes_data, tmp_path, monkeypatch) -> None:
    """A ``clips`` job over the float-bounded selection writes uint8 payloads.

    The full CLI→runner→writer path with the float ``(0.0, 255.0)`` value range
    and a ``clips`` algorithm is exactly where the adversarial run died: before
    the fix the payload writer refused the float64 output; after it, the trades
    are written and the bytes equal the (identity) source frames.
    """
    registry = dict(PluginLoadedRegistry.get_transformations_registry())
    registry["test_clips_identity"] = ClipsIdentityTransformation
    monkeypatch.setattr(PluginLoadedRegistry, "_transformations_registry", registry)

    root = tmp_path / "outputs"
    summary = run(
        build_config(
            loaders=[build_loader(parquet_path=raw_bytes_data)],
            transformations=[{"type": "test_clips_identity"}],
            output_root=root,
        )
    )
    assert summary == {"synthetic": 8}

    table = pq.read_table(str(root / "ledger" / "synthetic.parquet"))
    assert table.num_rows == 8
    assert table.column("algorithm").to_pylist() == ["test_clips_identity"] * 8
    assert table.column("family").to_pylist() == ["procedural"] * 8

    payloads = sorted((root / "synthetic").glob("*.png"))
    assert len(payloads) == 8
    assert all(np.asarray(Image.open(p)).dtype == np.uint8 for p in payloads)

    by_parent = dict(zip(table.column("parent_id").to_pylist(), table.column("artifact").to_pylist()))
    frames = raw_bytes_data.parent / "frames"
    for i in range(8):
        parent = f"syn_{i:04d}"
        source = (frames / f"img_{i:02d}.png").read_bytes()
        assert (root / "synthetic" / by_parent[parent]).read_bytes() == source


def test_clips_job_ledger_carries_null_seed_and_params(raw_bytes_data, tmp_path, monkeypatch) -> None:
    """The deterministic identity records the usual procedural row facts.

    A parameter-free, non-stochastic ``clips`` algorithm logs ``seed`` null,
    empty ``params``, ``reversible`` true, unary arity — the declarations the
    downstream judge reads off the ledger.
    """
    registry = dict(PluginLoadedRegistry.get_transformations_registry())
    registry["test_clips_identity"] = ClipsIdentityTransformation
    monkeypatch.setattr(PluginLoadedRegistry, "_transformations_registry", registry)

    root = tmp_path / "outputs"
    run(
        build_config(
            loaders=[build_loader(parquet_path=raw_bytes_data)],
            transformations=[{"type": "test_clips_identity"}],
            output_root=root,
        )
    )

    table = pq.read_table(str(root / "ledger" / "synthetic.parquet"))
    assert table.column("seed").to_pylist() == [None] * 8
    assert table.column("params").to_pylist() == ["{}"] * 8
    assert table.column("reversible").to_pylist() == [True] * 8
    assert table.column("arity").to_pylist() == ["unary"] * 8