"""Output configuration for the sample-generation interface.

The pipeline's output holds a **metadata-only** ledger plus, optionally,
hashed sample files in a payload store. The pixels never live in the parquet;
the key names stay generic (``samples_dir``/``write_samples``) because the
generic core configures the output writer, which may hold any datatype.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SamplingOutputsConfig(BaseModel):
    """Configuration of the sample-generation outputs.

    Attributes:
        path: Metadata-only ledger path, with ``{selection}`` substitutable by
            the selection name.
        write_samples: Emit hashed payload files; ``False`` runs a pure recipe
            trace with no payload files.
        samples_dir: Payload store: one encoded sample file per row, with
            ``{selection}`` substitutable by the selection name.
        payload_format: The registered payload writer that encodes each
            sample file (``images`` for PNG, ``npy`` for any array).
        flush_batch_size: Output rows buffered before a parquet write; also
            bounds the payload writer's in-memory buffer.
        include: Source columns passed through to the ledger, fnmatch wildcard
            ok; ``None`` passes nothing through (unless ``exclude`` is set,
            which passes everything except the excluded columns).
        exclude: Source columns withheld from the ledger; wins over
            ``include``.
        storage: Storage override (local only).
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Metadata-only ledger, {selection} substitutable.")
    write_samples: bool = Field(
        default=True,
        description="Emit hashed payload files; False → trace-only.",
    )
    samples_dir: str = Field(description="Payload store, {selection} substitutable.")
    payload_format: str = Field(
        default="images",
        description="Registered payload writer: 'images' (PNG) or 'npy' (any array).",
    )
    flush_batch_size: int = Field(
        default=5,
        description="Rows buffered before a parquet write; bounds the payload buffer.",
    )
    include: list[str] | None = Field(
        default=None,
        description="Source columns passed through to the ledger (wildcards ok); "
        "None passes nothing unless exclude is set.",
    )
    exclude: list[str] | None = Field(
        default=None,
        description="Source columns withheld from the ledger; wins over include.",
    )
    storage: bool | dict[str, Any] | None = Field(
        default=None,
        description="Storage configuration (local only).",
    )

    @field_validator("payload_format")
    @classmethod
    def _not_the_ledger(cls, v: str) -> str:
        """Refuse the ledger writer as a payload format.

        Args:
            v: The ``payload_format`` string from the config.

        Returns:
            The validated format.

        Raises:
            ValueError: If the format names the ``parquet`` ledger writer.
        """
        if v == "parquet":
            raise ValueError("payload_format 'parquet' is the ledger writer, not a payload format")
        return v
