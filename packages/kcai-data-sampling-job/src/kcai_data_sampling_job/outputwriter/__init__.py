"""Output writers package: payload + ledger plugins.

The generic ``OutputWriter`` contract lives in core
(``kcai_data_sampling_core.api.output_writer``); this package ships the
registered concrete writers — ``images`` and ``npy`` (payload files, chosen by
``outputs.payload_format``) and ``parquet`` (metadata-only ledger), registered
under the
``kcai_data_sampling.outputwriter`` entry-point group.
"""

from kcai_data_sampling_core.api.output_writer import OutputWriter

from kcai_data_sampling_job.outputwriter.api.images import ImagesOutputWriter
from kcai_data_sampling_job.outputwriter.api.npy import NpyOutputWriter
from kcai_data_sampling_job.outputwriter.api.parquet import ParquetOutputWriter

__all__ = ["ImagesOutputWriter", "NpyOutputWriter", "OutputWriter", "ParquetOutputWriter"]
