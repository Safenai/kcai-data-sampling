"""The generic output-writer base: payload files and ledger rows.

Two complementary writers cover the sample-generation outputs: a payload
writer encodes each generated sample to a hashed artifact file, and a ledger
writer holds the metadata-only table. Both subclass the generic
:class:`OutputWriter` here, each no-op'ing the role it does not own, and the
*concrete* writer configs (payload directory, ledger pattern) live with the
plugins in ``-job`` — core stays I/O-agnostic.
"""

from typing import Any


class OutputWriter:
    """Base writer for the two-step output surface.

    Generic methods are mostly no-ops: a subclass implements the role it
    owns (payload encoding, or ledger row buffering) and inherits the other.
    ``add_payload``/``add_rows`` only buffer in memory; no I/O happens until
    ``flush()``.

    Attributes:
        name: Unique writer name.
    """

    name: str

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Build the writer.

        Args:
            name: Unique writer name.
            config: Writer configuration; each concrete writer honors the
                keys it declares.
        """
        del config
        self.name = name

    def add_payload(self, selection_name: str, output: Any) -> str | None:
        """Encode one output to payload bytes; return its artifact name.

        Buffers in memory; no I/O happens until ``flush()``. Writers that own
        no payload (the ledger) return ``None``.

        Args:
            selection_name: The selection the output belongs to.
            output: An ``Output`` whose ``x`` carries the sample array.

        Returns:
            The payload artifact file name, or ``None``.
        """
        del selection_name, output
        return None

    def add_rows(self, selection_name: str, rows: list[dict]) -> None:
        """Buffer ledger rows for one selection.

        Args:
            selection_name: The selection the rows belong to.
            rows: Row dictionaries (the ledger columns).
        """
        del selection_name, rows

    def flush(self) -> None:
        """Write all buffered payloads and/or ledger rows.

        Called by the job at selection end, and by a writer itself once a
        threshold is reached. The base writes nothing.
        """