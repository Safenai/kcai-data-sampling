"""The two output writers' lifecycle and buffering.

The parquet writer must persist exactly what it was given, once, at ``flush``
(its single-file write is what makes the ledger batch-invariant); the images
writer buffers encoded payloads in memory, flushes at ``flush_batch_size``,
and in trace-only mode returns the content-hash artifact name without touching
the disk.
"""

import hashlib
import re

from kcai_data_sampling_images.api.transformations.horizontal_flip import HorizontalFlip
from kcai_data_sampling_job.outputwriter.api.images import ImagesOutputWriter
from kcai_data_sampling_job.outputwriter.api.parquet import ParquetOutputWriter
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import pytest

ARTIFACT_RE = re.compile(r"^(?P<selection>.+)__(?P<id>[0-9a-f]{12})__(?P<c6>[0-9a-f]{6})\.png$")


def _outputs(synthetic_batch):
    """One Output per synthetic row, via a real transformation."""
    return HorizontalFlip({}).transform(synthetic_batch, synthetic_batch.value_range)


def test_parquet_writer_writes_once_on_flush(tmp_path) -> None:
    """Ledger rows land in one parquet file at flush, whatever add_rows calls.

    The writer buffers across ``add_rows`` and writes a single table at
    ``flush`` — the mechanism behind the ledger being byte-identical for any
    flush batch size (batch invariance).
    """
    writer = ParquetOutputWriter(
        name="ledger",
        config={"path_pattern": str(tmp_path / "ledger" / "{selection}.parquet")},
    )
    writer.add_rows("synthetic", [{"parent_id": "a", "id": "111111111111"}, {"parent_id": "b", "id": "222222222222"}])
    writer.add_rows("synthetic", [{"parent_id": "c", "id": "333333333333"}])
    writer.flush()

    table = pq.read_table(tmp_path / "ledger" / "synthetic.parquet")
    assert table.column_names == ["parent_id", "id"]
    assert table.num_rows == 3
    assert table.column("parent_id").to_pylist() == ["a", "b", "c"]


def test_parquet_writer_second_selection_writes_a_fresh_file(tmp_path) -> None:
    """After a flush, a new selection gets its own file (fresh buffer state).

    Each selection's path is fixed at the first added batch and the buffer is
    cleared on flush, so consecutive selections never leak rows into each
    other's ledger.
    """
    writer = ParquetOutputWriter(
        name="ledger",
        config={"path_pattern": str(tmp_path / "ledger" / "{selection}.parquet")},
    )
    writer.add_rows("a", [{"id": "000000000001"}])
    writer.flush()
    writer.add_rows("b", [{"id": "000000000002"}])
    writer.flush()

    assert pq.read_table(tmp_path / "ledger" / "a.parquet").num_rows == 1
    assert pq.read_table(tmp_path / "ledger" / "b.parquet").num_rows == 1


def test_parquet_writer_requires_a_path_pattern() -> None:
    """A ledger writer without a ``path_pattern`` is misconfigured at build."""
    with pytest.raises(ValueError, match="requires a 'path_pattern'"):
        ParquetOutputWriter(name="ledger", config={})


def test_images_writer_flushes_at_the_threshold_buffer_size(tmp_path, synthetic_batch) -> None:
    """Payloads buffer in memory and flush exactly at flush_batch_size.

    With a threshold of 2, the first encode stays buffered (nothing on disk),
    the second triggers a flush (two files appear), and the final flush writes
    the rest — the writer never holds more than the bound, and flushing
    an empty buffer is a no-op.
    """
    writer = ImagesOutputWriter(
        name="images",
        config={"samples_dir": str(tmp_path / "{selection}"), "write_samples": True, "flush_batch_size": 2},
    )
    outputs = _outputs(synthetic_batch)

    writer.add_payload("synthetic", outputs[0])
    assert len(list((tmp_path / "synthetic").glob("*.png"))) == 0
    writer.add_payload("synthetic", outputs[1])
    assert len(list((tmp_path / "synthetic").glob("*.png"))) == 2
    writer.add_payload("synthetic", outputs[2])
    assert len(list((tmp_path / "synthetic").glob("*.png"))) == 2
    writer.flush()
    writer.flush()
    assert len(list((tmp_path / "synthetic").glob("*.png"))) == 3


def test_images_writer_pngs_decode_back_to_the_expected_pixels(tmp_path, synthetic_batch) -> None:
    """A written payload PNG round-trips to the transformed (H, W, 4) row.

    The stored bytes are the flipped row: opening the file gives the same
    widths, height, channels and value range the transformation produced.
    """
    writer = ImagesOutputWriter(
        name="images",
        config={"samples_dir": str(tmp_path / "{selection}"), "write_samples": True, "flush_batch_size": 100},
    )
    outputs = _outputs(synthetic_batch)
    names = [writer.add_payload("synthetic", o) for o in outputs]
    writer.flush()

    for name, output in zip(names, outputs, strict=True):
        restored = np.asarray(Image.open(tmp_path / "synthetic" / name))
        assert restored.shape == output.x.shape
        assert restored.dtype == np.uint8
        assert restored.min() >= 0
        assert restored.max() <= 255
        assert np.array_equal(restored, output.x)


def test_images_writer_artifact_name_is_content_addressed(tmp_path, synthetic_batch) -> None:
    """Artifact names are ``selection__id__c6.png`` with c6 hashing the pixels.

    The six-hex suffix verifiably hashes the decoded payload array, so a file
    can be re-derived from the row: identical pixels plus the same
    output id always name the same file.
    """
    writer = ImagesOutputWriter(
        name="images",
        config={"samples_dir": str(tmp_path / "{selection}"), "write_samples": True, "flush_batch_size": 100},
    )
    outputs = _outputs(synthetic_batch)
    names = [writer.add_payload("synthetic", o) for o in outputs]
    writer.flush()

    for name, output in zip(names, outputs, strict=True):
        match = ARTIFACT_RE.fullmatch(name)
        assert match is not None
        assert match.group("selection") == "synthetic"
        assert match.group("id") == output.id
        assert match.group("c6") == hashlib.sha1(output.x.tobytes()).hexdigest()[:6]
        assert (tmp_path / "synthetic" / name).exists


def test_images_writer_trace_only_writes_no_files(tmp_path, synthetic_batch) -> None:
    """Trace-only mode returns artifact hashes without touching the disk.

    ``write_samples: false`` means a pure recipe trace: every output still gets
    its content-addressed name, but nothing is buffered or written, so no
    payload directory appears at all.
    """
    writer = ImagesOutputWriter(
        name="images",
        config={"samples_dir": str(tmp_path / "{selection}"), "write_samples": False},
    )
    outputs = _outputs(synthetic_batch)
    names = [writer.add_payload("synthetic", o) for o in outputs]

    assert len(names) == len(outputs)
    assert all(ARTIFACT_RE.fullmatch(n) is not None for n in names)
    assert not (tmp_path / "synthetic").exists()
