"""Loader subclasses: the ``parquet`` image and ``timeseries_parquet`` loaders.

The generic loader contracts live in core; this folder holds the job
package's dataloader subclasses and their plugin-owned config schemas.
"""

from kcai_data_sampling_job.dataloaders.api.parquet import (
    ParquetDataLoader,
    ParquetDataSelection,
    ParquetImageLoaderConfig,
)
from kcai_data_sampling_job.dataloaders.api.timeseries import (
    ConsecutiveTimeSeriesSamples,
    IndependentTimeSeriesSamples,
    TimeSeriesLoaderConfig,
    TimeSeriesParquetDataLoader,
)

__all__ = [
    "ConsecutiveTimeSeriesSamples",
    "IndependentTimeSeriesSamples",
    "ParquetDataLoader",
    "ParquetDataSelection",
    "ParquetImageLoaderConfig",
    "TimeSeriesLoaderConfig",
    "TimeSeriesParquetDataLoader",
]
