"""Pipeline orchestration, YAML CLI, and generic table IO.

The ``parquet`` column loader first decodes `img_bytes` rows to in-memory
samples, then the transformation interface runs over them; the ``parquet``
output writer owns the metadata-only ledger. The image sample readers/writers
are plugins shipped by ``kcai-data-sampling-images``.
"""

from kcai_data_sampling_job._version_ import __version__ as __version__
