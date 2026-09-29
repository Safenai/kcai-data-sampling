"""Image loader subclasses: the ``parquet`` dataloader.

The generic loader contracts live in core; this folder holds the image
package's dataloader subclasses and their plugin-owned config schema.
"""

from kcai_data_sampling_job.dataloaders.api.parquet import (
    ParquetDataLoader,
    ParquetDataSelection,
    ParquetImageLoaderConfig,
)

__all__ = ["ParquetDataLoader", "ParquetDataSelection", "ParquetImageLoaderConfig"]
