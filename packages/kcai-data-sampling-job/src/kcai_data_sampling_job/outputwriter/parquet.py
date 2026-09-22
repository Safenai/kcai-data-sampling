"""The ``parquet`` output writer: the metadata-only ledger.

Accumulates the rows of one selection and writes a single parquet file at
``flush()``, so the written ledger never depends on ``flush_batch_size``
(batch invariance). Rows carry no pixels: each references its artifact
file in ``images_dir``.
"""

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ParquetOutputWriter:
    """Ledger writer for the generated-sample rows.

    The plugin for the ``parquet`` registered type
    (``kcai_data_sampling.outputwriter`` entry point). ``write_payload`` is a
    no-op: this writer owns the ledger, not the pixels.

    Attributes:
        name: Unique writer name.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Build the ledger writer.

        Args:
            name: Unique writer name.
            config: Writer configuration; ``path_pattern`` (ledger path with
                ``{selection}`` substitutable) and ``flush_batch_size``
                (accepted, buffering hint — the file is written once) are
                honored.

        Raises:
            ValueError: If ``path_pattern`` is missing.
        """
        config = config or {}
        if "path_pattern" not in config:
            raise ValueError("parquet output writer requires a 'path_pattern'")
        self.name = name
        self.path_pattern = config["path_pattern"]
        self.file_path: Path | None = None
        self._rows: list[dict[str, Any]] = []

    def write_payload(self, selection_name: str, output: Any) -> None:
        """No payload files: the ledger is metadata-only.

        Args:
            selection_name: Ignored.
            output: Ignored.
        """
        del selection_name, output

    def add_rows(self, selection_name: str, rows: list[dict]) -> None:
        """Buffer ledger rows for one selection.

        Args:
            selection_name: The selection the rows belong to; its ledger file
                name is fixed at the first added batch.
            rows: The row dictionaries.
        """
        if self.file_path is None:
            self.file_path = Path(self.path_pattern.format(selection=selection_name))
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._rows.extend(rows)

    def flush(self) -> None:
        """Write all buffered rows to the ledger, once, then reset.

        A single ``write_table`` is what makes the ledger byte-identical
        whatever the flush batch size. Does nothing when no row was
        buffered.
        """
        if not self._rows or self.file_path is None:
            return
        table = pa.Table.from_pylist(self._rows)
        pq.write_table(table, self.file_path)
        logger.info("Wrote ledger with %s rows to %s", len(self._rows), self.file_path)
        self._rows = []
        self.file_path = None