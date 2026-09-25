"""Output writers package: payload + ledger plugins.

The generic ``OutputWriter`` contract lives in core
(``kcai_data_sampling_core.api.output_writer``); this package ships the two
registered concrete writers — ``images`` (payload files) and ``parquet``
(metadata-only ledger), registered under the
``kcai_data_sampling.outputwriter`` entry-point group.
"""

from kcai_data_sampling_core.api.output_writer import OutputWriter
from kcai_data_sampling_job.outputwriter.api.images import ImagesOutputWriter
from kcai_data_sampling_job.outputwriter.api.parquet import ParquetOutputWriter

__all__ = ["OutputWriter", "ImagesOutputWriter", "ParquetOutputWriter"]