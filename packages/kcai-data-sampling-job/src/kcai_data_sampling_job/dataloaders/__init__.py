"""Data loaders package: the parquet image and time-series loaders.

``image_dir`` has been dropped (parquet is the only input); this package
ships the image ``parquet`` table loader and the ``timeseries_parquet`` series
loader under ``api/`` (the subclasses folder for this package), registered
under the ``kcai_data_sampling.dataloaders`` entry-point group. The generic
``DataLoader``/``DataSelection`` protocols live in core
(``kcai_data_sampling_core.api.dataloaders``).
"""

from kcai_data_sampling_core.api.dataloaders import DataLoader, DataSelection

from kcai_data_sampling_job.dataloaders.api.parquet import (
    ParquetDataLoader,
    ParquetDataSelection,
    ParquetImageLoaderConfig,
)
from kcai_data_sampling_job.dataloaders.api.timeseries import (
    ConsecutiveTimeSeriesSamples,
    IndependentTimeSeriesSamples,
    TimeSeriesParquetDataLoader,
    TimeSeriesLoaderConfig,
)

__all__ = [
    "ConsecutiveTimeSeriesSamples",
    "DataLoader",
    "DataSelection",
    "IndependentTimeSeriesSamples",
    "ParquetDataLoader",
    "ParquetDataSelection",
    "ParquetImageLoaderConfig",
    "TimeSeriesLoaderConfig",
    "TimeSeriesParquetDataLoader",
]
