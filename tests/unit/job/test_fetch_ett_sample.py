"""The ETT fetch script writes an explicit, statistics-rich parquet schema."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "fetch_ett_sample.py"
CSV = (
    "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n2016-07-01 00:00:00,1,2,3,4,5,6,7\n2016-07-01 01:00:00,8,9,10,11,12,13,14\n"
)


def _module() -> ModuleType:
    """Load the script without running its CLI."""
    spec = importlib.util.spec_from_file_location("fetch_ett_sample", SCRIPT)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_fetch_script_writes_explicit_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The written file has fixed channel lists, `n_steps`, zstd and statistics."""
    module = _module()
    monkeypatch.setattr(module, "fetch_csv", lambda variant: CSV)
    monkeypatch.setattr(sys, "argv", ["fetch_ett_sample.py", "--variants", "ETTh1", "--out", str(tmp_path)])
    module.main()

    parquet = pq.ParquetFile(tmp_path / "series.parquet")
    schema = parquet.schema_arrow
    assert schema.names == ["series_id", "values", "n_steps", "start", "freq"]
    assert schema.field("values").type == pa.list_(pa.list_(pa.float64(), 7))
    assert schema.field("n_steps").type == pa.int64()
    row_group = parquet.metadata.row_group(0)
    assert row_group.column(1).compression == "ZSTD"
    assert row_group.column(1).statistics is not None
    assert parquet.read().column("n_steps").to_pylist() == [2]
