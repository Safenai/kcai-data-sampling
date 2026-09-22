"""Output configuration for the sample-generation interface (NEW).

The image pipeline's output holds a **metadata-only** ledger plus, optionally,
hashed image files in a payload store. The pixels never live in the parquet.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SamplingOutputsConfig(BaseModel):
    """Configuration of the sample-generation outputs.

    Attributes:
        path: Metadata-only ledger path, with ``{selection}`` substitutable by
            the selection name.
        write_images: Emit hashed payload files; ``False`` runs a pure recipe
            trace with no payload files.
        images_dir: Payload store: one encoded image file per row, with
            ``{selection}`` substitutable by the selection name.
        flush_batch_size: Output rows buffered before a parquet write.
        storage: Storage override (local only this phase).
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Metadata-only ledger, {selection} substitutable.")
    write_images: bool = Field(default=True, description="Emit hashed payload files; False → trace-only.")
    images_dir: str = Field(description="Payload store, {selection} substitutable.")
    flush_batch_size: int = Field(default=128, description="Output rows buffered before a parquet write.")
    storage: bool | dict[str, Any] | None = Field(
        default=None,
        description="Storage configuration (local only this phase).",
    )