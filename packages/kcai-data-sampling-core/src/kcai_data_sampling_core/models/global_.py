"""Global configuration models: storage, compute, and error handling.

Composed from dqm-ml's ``global_.py`` with S3 support deliberately cut:
storage is local only, but the ``type`` discriminator keeps the same
shape so S3 can be re-added later behind the same interface.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StorageConfig(BaseModel):
    """Local storage configuration.

    Local only. The ``type`` field
    keeps the discriminator shape so an S3 backend can be re-added behind the
    same interface.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["local"] = Field(default="local", description="Storage backend type.")


class ComputeConfig(BaseModel):
    """Global compute / runtime settings.

    Attributes:
        seed: Random seed for reproducibility.
        log_level: Logging verbosity.
        max_memory: Maximum memory per worker (e.g. ``"4Gi"``), used as a
            memory-flush threshold.
        device: Compute device: ``"auto"`` picks cuda if available (GPU paths
            arrive with a parallel runner).
        progress_bar: Show tqdm progress bars.
        threads: Number of worker threads for IO.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, description="Random seed for reproducibility.")
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="warning",
        description="Logging verbosity.",
    )
    max_memory: str | None = Field(default=None, description="Maximum memory per worker (e.g. '4Gi').")
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Compute device: 'auto' picks cuda if available.",
    )
    progress_bar: bool = Field(default=True, description="Show tqdm progress bars.")
    threads: int = Field(default=4, gt=0, description="Number of worker threads.")


class ImageErrorsConfig(BaseModel):
    """Error-handling policy for image-processing failures.

    Attributes:
        on_decode_failure: What to do when an image cannot be decoded.
        on_transform_error: What to do when a transform fails.
        on_unsupported_format: What to do on an unsupported image format.
    """

    model_config = ConfigDict(extra="forbid")

    on_decode_failure: Literal["silent_fail", "fail_fast"] = Field(
        default="silent_fail",
        description="Action when an image cannot be decoded.",
    )
    on_transform_error: Literal["silent_fail", "fail_fast"] = Field(
        default="silent_fail",
        description="Action when an image transform fails.",
    )
    on_unsupported_format: Literal["silent_fail", "fail_fast"] = Field(
        default="fail_fast",
        description="Action on unsupported image format.",
    )


class TabularErrorsConfig(BaseModel):
    """Error-handling policy for tabular-data failures.

    Attributes:
        on_missing_column: What to do when a required column is missing.
        on_file_not_found: What to do when a data file is not found.
    """

    model_config = ConfigDict(extra="forbid")

    on_missing_column: Literal["silent_fail", "fail_fast"] = Field(
        default="fail_fast",
        description="Action when a required column is missing.",
    )
    on_file_not_found: Literal["silent_fail", "fail_fast"] = Field(
        default="fail_fast",
        description="Action when a data file is not found.",
    )


class ErrorsConfig(BaseModel):
    """Aggregate error-handling configuration.

    Attributes:
        default: Default action when no specific policy is set.
        images: Policy for image-processing failures.
        tabular: Policy for tabular-data failures.
        max_failure_rate: Maximum tolerated failure rate (0.0 to 1.0) before
            the job aborts.
    """

    model_config = ConfigDict(extra="forbid")

    default: Literal["silent_fail", "fail_fast"] = Field(
        default="silent_fail",
        description="Default error action when no specific policy is set.",
    )
    images: ImageErrorsConfig | None = None
    tabular: TabularErrorsConfig | None = None
    max_failure_rate: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Maximum tolerated failure rate before the job aborts (from 0.0 to 1.0).",
    )