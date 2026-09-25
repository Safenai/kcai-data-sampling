"""The payload output writer: one hashed PNG payload file per row.

Writes the pixel side of the metadata-only ledger:
``write_samples: false`` turns the run into a pure recipe trace (hashes only),
the payload files hold the pixels.

Payload rows are encoded to bytes in memory and buffered; ``flush()`` writes
them. ``flush_batch_size`` bounds the buffer, so the writer never holds more
than that many encoded rows in memory.
"""

from pathlib import Path

from typing_extensions import override

from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_core.api.output_writer import OutputWriter
from kcai_data_sampling_job.utils.images import artifact_name, png_bytes


class ImagesOutputWriter(OutputWriter):
    """Writes one encoded image file per output row.

    The plugin for the ``images`` registered type
    (``kcai_data_sampling.outputwriter`` entry point). It owns the payload
    files only; the parquet writer owns the ledger, so this writer inherits
    the common writer protocol's no-op ``add_rows``.

    Attributes:
        name: Unique writer name.
    """

    def __init__(self, name: str, config: dict | None = None):
        """Build the payload writer.

        Args:
            name: Unique writer name.
            config: Writer configuration; ``samples_dir`` (payload directory
                pattern, ``{selection}`` substitutable), ``write_samples``
                (emit files or trace-only) and ``flush_batch_size`` (payload
                rows buffered before a flush) are honored.
        """
        config = config or {}
        super().__init__(name, config)
        self.samples_dir = config.get("samples_dir", "outputs/image/")
        self.write_samples = bool(config.get("write_samples", True))
        self.flush_batch_size = config.get("flush_batch_size", 5)
        self._buffer: list[tuple[Path, bytes]] = []

    @override
    def add_payload(self, selection_name: str, output: Output) -> str | None:
        """Encode one output row to PNG bytes and buffer it.

        The artifact name is content-addressed, so buffering and flush order
        cannot change the pixels on disk. No file is written until ``flush()``.

        Args:
            selection_name: The selection the row belongs to; substituted
                into the ``samples_dir`` pattern.
            output: The output row; its ``x`` is the ``(H, W, 4)`` uint8 array.

        Returns:
            The artifact file name (``{selection}__{id}__{c6}.png``). In
            trace-only mode the name is still returned (hash-only trace) but
            nothing is buffered.
        """
        artifact = artifact_name(
            selection_name=selection_name,
            output_id=output.id,
            x=output.x,
        )
        if self.write_samples:
            samples_dir = Path(self.samples_dir.format(selection=selection_name))
            self._buffer.append((samples_dir / artifact, png_bytes(output.x)))
            if len(self._buffer) >= self.flush_batch_size:
                self.flush()
        return artifact

    @override
    def add_rows(self, selection_name: str, rows: list[dict]) -> None:
        """Ledger rows belong to the parquet writer; nothing to do here.

        Args:
            selection_name: Ignored.
            rows: Ignored.
        """
        del selection_name, rows

    @override
    def flush(self) -> None:
        """Write all buffered payloads, then clear the buffer.

        Payload files are written synchronously here — called by the job at
        selection end, and by the writer itself once the threshold is reached.
        """
        for path, data in self._buffer:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self._buffer.clear()