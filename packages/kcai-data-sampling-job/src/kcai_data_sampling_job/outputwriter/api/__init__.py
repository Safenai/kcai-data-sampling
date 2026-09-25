"""Output-writer subclasses of the generic core base.

The generic ``OutputWriter`` contract lives in core
(``kcai_data_sampling_core.api.output_writer``); this folder holds the
package's concrete writers — the payload plugin and the ledger writer — each
implementing the role it owns and inheriting the other as a no-op.
"""

from kcai_data_sampling_job.outputwriter.api.images import ImagesOutputWriter
from kcai_data_sampling_job.outputwriter.api.parquet import ParquetOutputWriter

__all__ = ["ImagesOutputWriter", "ParquetOutputWriter"]