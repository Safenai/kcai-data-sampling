"""Data loading configuration models.

Composed from dqm-ml's ``dataloaders.py`` with the batch-size rename the
three-batch-level design requires: ``batch_size`` is ``load_batch_size`` here,
the number of rows pulled from a selection per step.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kcai_data_sampling_core.models.global_ import StorageConfig


class FilterConfig(BaseModel):
    """Row-level filter applied during data loading.

    Attributes:
        column: Column name to filter on.
        values: Values to keep; rows where the column matches are included.
    """

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column name to filter on.")
    values: list[bool] | list[str] | list[int] | list[float] = Field(
        description="Value(s) to keep. Rows where column matches are included.",
    )


class SamplePathConfig(BaseModel):
    """Per-column path prefix configuration.

    Attributes:
        column: Column name containing relative file paths.
        prefix: Base directory for resolving relative paths.
    """

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column name containing relative file paths.")
    prefix: str | None = Field(default=None, description="Base directory for resolving relative paths.")


class SplitConfig(BaseModel):
    """How to split data into named groups (e.g. train / test).

    Attributes:
        by: Column used to determine the split group.
        values: Explicit list of split-group values to materialise;
            auto-discovered if ``None``.
        exclude: Split-group values to exclude (fnmatch patterns supported).
    """

    model_config = ConfigDict(extra="forbid")

    by: str = Field(description="Column used to determine the split group.")
    values: list[str] | None = Field(
        default=None,
        description="Explicit list of split-group values to materialise. Auto-discovered if None.",
    )
    exclude: list[str] | None = Field(
        default=None,
        description="Split-group values to exclude (fnmatch patterns supported).",
    )


class TransformType(str, Enum):
    """Target data type for column transformations."""

    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BOOL = "bool"
    STR = "str"
    CATEGORICAL = "categorical"


class TransformConfig(BaseModel):
    """Column type-casting transformation.

    Attributes:
        column: Column name to transform.
        to_type: Target data type.
        in_place: Overwrite the original column in place.
    """

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column name to transform.")
    to_type: TransformType = Field(description="Target data type.")
    in_place: bool = Field(default=False, description="Overwrite the original column in place.")


class DataLoaderConfig(BaseModel):
    """Configuration for a single dataloader.

    ``type`` is registry-resolved: the loader plugin
    classes are discovered from entry points, so a new loader package plugs in
    without touching core.

    Attributes:
        name: Unique name for this dataloader.
        type: Registered dataloader type (``image_dir``, ``parquet``, ...).
        path: Local path or glob pattern to the data files.
        id_column: Column (or file stem) used as row identifier.
        load_batch_size: Rows pulled from the selection per step.
        decode: Image mode knob; only ``img_bytes`` is implemented this phase
            (``rgba`` input is postponed).
        image_shape: Optional ``[height, width, channels]`` for image tables
            whose per-row shape is not carried in the data.
        filters: Row-level filters.
        sample_path: Per-column path prefixes.
        split: Group the data into named selections.
        transform: Column type-casting transforms.
        storage: Storage override (local only this phase).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique name for this dataloader.")
    type: str = Field(description="Registered dataloader type (plugin).")
    path: str = Field(description="Glob pattern or path to data files.")
    id_column: str | None = Field(default=None, description="Column used as row identifier.")
    load_batch_size: int = Field(default=10000, description="Number of rows per step.")
    decode: Literal["img_bytes", "rgba"] | None = Field(
        default=None,
        description="Image payload form; only 'img_bytes' is implemented this phase.",
    )
    image_shape: list[int] | None = Field(
        default=None,
        description="Image shape [height, width, channels] when not carried in the data.",
    )
    filters: list[FilterConfig] | None = None
    sample_path: list[SamplePathConfig] | None = None
    split: SplitConfig | None = None
    transform: list[TransformConfig] | None = None
    storage: StorageConfig | None = None


class DataLoadersConfig(BaseModel):
    """Collection of dataloaders for a job.

    Attributes:
        storage: Default storage config inherited by all loaders.
        loaders: List of dataloader configurations.
    """

    model_config = ConfigDict(extra="forbid")

    storage: StorageConfig | None = Field(
        default=None,
        description="Default storage config inherited by all loaders.",
    )
    loaders: list[DataLoaderConfig] = Field(description="List of dataloader configurations.")