"""Output writers: the metadata ledger (parquet) and the payload plugin.

Two complementary writers cover the sample-generation outputs: a payload
writer encodes each generated image to a hashed artifact file (the ``images``
plugin), and a ledger writer holds the metadata-only parquet. Both conform to
the :class:`OutputWriter` protocol below, each no-oping the role it does not own
in the protocol's write surface.
"""

from typing import Any, Protocol, runtime_checkable

from kcai_data_sampling_job.outputwriter.parquet import ParquetOutputWriter


@runtime_checkable
class OutputWriter(Protocol):
    """Protocol for the two-phase output writing.

    Attributes:
        name: Unique writer name.
    """

    name: str

    def write_payload(self, selection_name: str, output: Any) -> str | None:
        """Encode one output to a payload file; return its artifact name.

        Args:
            selection_name: The selection the output belongs to.
            output: An ``Output`` whose ``x`` carries the pixel array. The
                ledger writer returns ``None`` (no payload).

        Returns:
            The payload artifact file name, or ``None``.
        """

    def add_rows(self, selection_name: str, rows: list[dict]) -> None:
        """Buffer ledger rows for one selection.

        Args:
            selection_name: The selection the rows belong to.
            rows: Row dictionaries (the ledger columns).
        """

    def flush(self) -> None:
        """Write all buffered ledger rows (once, at selection end)."""


__all__ = ["OutputWriter", "ParquetOutputWriter"]