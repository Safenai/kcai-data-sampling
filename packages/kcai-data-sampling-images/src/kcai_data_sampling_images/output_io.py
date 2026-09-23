"""The ``images`` output writer: one hashed PNG payload file per row.

Writes the pixel side of the metadata-only ledger:
``write_images: false`` turns the run into a pure recipe trace (hashes only),
the payload files hold the pixels.
"""

from pathlib import Path

from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_images.utils.images import artifact_name, encode_image


class ImagesOutputWriter:
    """Writes one encoded image file per output row.

    The plugin for the ``images`` registered type
    (``kcai_data_sampling.outputwriter`` entry point). It owns the payload
    files only; the parquet writer owns the ledger, so this writer's
    ``add_rows``/``flush`` are no-ops of the common writer protocol.

    Attributes:
        name: Unique writer name.
    """

    def __init__(self, name: str, config: dict | None = None):
        """Build the payload writer.

        Args:
            name: Unique writer name.
            config: Writer configuration; ``images_dir`` (payload directory
                pattern, ``{selection}`` substitutable) and ``write_images``
                (emit files or trace-only) are honored.
        """
        config = config or {}
        self.name = name
        self.images_dir = config.get("images_dir", "outputs/image/")
        self.write_images = bool(config.get("write_images", True))

    def write_payload(self, selection_name: str, output: Output) -> str | None:
        """Encode one output row and return its artifact file name.

        Args:
            selection_name: The selection the row belongs to; substituted
                into the ``images_dir`` pattern.
            output: The output row; its ``x`` is the ``(H, W, 4)`` uint8 array.

        Returns:
            The artifact file name (``{selection}__{id}__{c6}.png``), or
            ``None`` in trace-only mode.
        """
        artifact = artifact_name(
            selection_name=selection_name,
            output_id=output.id,
            x=output.x,
        )
        if self.write_images:
            images_dir = Path(self.images_dir.format(selection=selection_name))
            encode_image(images_dir / artifact, output.x)
        return artifact

    def add_rows(self, selection_name: str, rows: list[dict]) -> None:
        """Ledger rows belong to the parquet writer; nothing to do here.

        Args:
            selection_name: Ignored.
            rows: Ignored.
        """
        del selection_name, rows

    def flush(self) -> None:
        """Nothing buffered; payload files are written synchronously."""