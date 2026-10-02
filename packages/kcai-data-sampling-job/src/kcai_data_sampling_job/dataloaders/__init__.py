"""Data loaders package: the parquet image loader and its plugin config.

``image_dir`` has been dropped (parquet is the only input); this package
ships the image ``parquet`` table loader under ``api/`` (the subclasses folder
for this package), registered under the ``kcai_data_sampling.dataloaders``
entry-point group. The generic ``DataLoader``/``DataSelection`` protocols live
in core (``kcai_data_sampling_core.api.dataloaders``).
"""

from kcai_data_sampling_core.api.dataloaders import DataLoader, DataSelection

from kcai_data_sampling_job.dataloaders.api.parquet import (
    ParquetDataLoader,
    ParquetDataSelection,
    ParquetImageLoaderConfig,
)

__all__ = [
    "DataLoader",
    "DataSelection",
    "ParquetDataLoader",
    "ParquetDataSelection",
    "ParquetImageLoaderConfig",
]
