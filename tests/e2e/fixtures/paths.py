"""Where an e2e job's products land: output-tree path helpers.

Every e2e test runs a job into a per-test ``tmp_path``; these helpers say where
the ledger parquet and the payload store end up for a given selection, so the
tests assert on the real on-disk surface the CLI produced rather than on
writer internals.
"""

from __future__ import annotations

from pathlib import Path

#: The selection name every e2e run uses (one selection per loader).
SELECTION = "synthetic"


def output_root(tmp_path: Path) -> Path:
    """The root directory a test's job writes under.

    Args:
        tmp_path: Pytest per-test directory.

    Returns:
        ``tmp_path/outputs`` — the ``output_root`` passed to the config
        builders.
    """
    return tmp_path / "outputs"


def ledger_file(root: Path, selection: str = SELECTION) -> Path:
    """The parquet ledger path for one selection under the output root.

    Args:
        root: The job's output root (see ``output_root``).
        selection: Selection name; the ledger file is ``ledger/<selection>.parquet``.

    Returns:
        The ledger parquet file path.
    """
    return root / "ledger" / f"{selection}.parquet"


def samples_dir(root: Path, selection: str = SELECTION) -> Path:
    """The payload store (hashed PNG directory) for one selection.

    Args:
        root: The job's output root (see ``output_root``).
        selection: Selection name; payloads live under ``<selection>/``.

    Returns:
        The payload directory path.
    """
    return root / selection