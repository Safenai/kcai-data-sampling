"""Data loaders module: protocol plus the parquet table loader.

``image_dir`` (folder of image files) is shipped by ``kcai-data-sampling-images``;
this package adds the generic ``parquet`` table loader. Both are registered
under the ``kcai_data_sampling.dataloaders`` entry-point group.
"""

from kcai_data_sampling_job.dataloaders.parquet import ParquetDataLoader
from kcai_data_sampling_job.dataloaders.proto import DataLoader, DataSelection

__all__ = ["DataLoader", "DataSelection", "ParquetDataLoader"]