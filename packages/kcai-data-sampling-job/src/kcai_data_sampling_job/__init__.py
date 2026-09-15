"""kcai-data-sampling-job: disk. Depends on the core; the core never depends on it."""

from kcai_data_sampling_job.store import Store

__all__ = ["Store"]
