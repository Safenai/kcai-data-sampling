"""Row-count and correctness checks for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def check_row_count(artifact_dir: Path, expected: int) -> bool:
    """Verify the ledger parquet has the expected number of rows.

    Args:
        artifact_dir: The run's artifact directory.
        expected: Expected row count.

    Returns:
        True if the count matches.
    """
    import pyarrow.parquet as pq

    ledger = artifact_dir / "ledger.parquet"
    if not ledger.exists():
        return False
    table = pq.read_table(ledger)
    return table.num_rows == expected


def check_determinism(run1: dict[str, Any], run2: dict[str, Any]) -> bool:
    """Check if two runs produce identical ids and hashes.

    Args:
        run1: First run's ledger data.
        run2: Second run's ledger data.

    Returns:
        True if they match.
    """
    return run1.get("ids") == run2.get("ids") and run1.get("hashes") == run2.get("hashes")


def check_equivalence(path: Path) -> bool:
    """Verify B2-b equivalence: 40/40 levels match.

    Args:
        path: Path to equivalence.csv.

    Returns:
        True if all levels are equal.
    """
    import csv

    if not path.exists():
        return False
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("equal", "").lower() != "true":
                return False
    return True
