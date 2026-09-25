"""Data loading configuration models.

Composed from dqm-ml's ``dataloaders.py`` with the batch-size rename the
three-batch-level design requires: ``batch_size`` is ``load_batch_size`` here,
the number of rows pulled from a selection per step.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    ``type`` is registry-resolved: the loader plugin classes are discovered
    from entry points, and each plugin carries its own config schema
    (``loader_cls.Config``, a subclass of this generic model) with the
    datatype-specific keys — so a new loader package plugs in without touching
    core.

    Attributes:
        name: Unique name for this dataloader.
        type: Registered dataloader type (``parquet``, ...).
        path: Local path or glob pattern to the data files.
        id_column: Column (or file stem) used as row identifier.
        load_batch_size: Rows pulled from the selection per step.
        filters: Row-level filters.
        sample_path: Per-column path prefixes.
        transform: Column type-casting transforms.
        storage: Storage override (local only).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique name for this dataloader.")
    type: str = Field(description="Registered dataloader type (plugin).")
    path: str = Field(description="Glob pattern or path to data files.")
    id_column: str | None = Field(default=None, description="Column used as row identifier.")
    load_batch_size: int = Field(default=10000, description="Number of rows per step.")
    filters: list[FilterConfig] | None = None
    sample_path: list[SamplePathConfig] | None = None
    transform: list[TransformConfig] | None = None
    storage: StorageConfig | None = None


class DataLoadersConfig(BaseModel):
    """Collection of dataloaders for a job.

    Each entry is validated against its loader plugin's own config schema
    (``loader_cls.Config``), mirroring how transformation entries are resolved
    against the algorithm's registered schema.

    Attributes:
        storage: Default storage config inherited by all loaders.
        loaders: List of raw dataloader configuration dicts, resolved through
            the loader registry.
    """

    model_config = ConfigDict(extra="forbid")

    storage: StorageConfig | None = Field(
        default=None,
        description="Default storage config inherited by all loaders.",
    )
    loaders: list[dict] = Field(description="List of raw dataloader configuration dicts.")

    @model_validator(mode="after")
    def _resolve_loaders(self) -> "DataLoadersConfig":
        """Resolve each raw loader entry against its registered schema.

        Each entry's ``type`` is looked up in the dataloader registry; the
        loader's own config schema (``loader_cls.Config``) validates the
        entry, so datatype-specific keys and required parameters are refused
        or kept at config load.

        Returns:
            This config with every loader validated against its specific
            registered schema.

        Raises:
            ValueError: If a loader type is unknown or an entry does not pass
                its loader's schema.
        """
        from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry

        registry = PluginLoadedRegistry.get_dataloaders_registry()
        resolved: list[DataLoaderConfig] = []
        for entry in self.loaders:
            entry = dict(entry)
            loader_cls = registry.get(entry.get("type", ""))
            if loader_cls is None:
                raise ValueError(f"unknown dataloader type {entry.get('type')!r}")
            config_cls = getattr(loader_cls, "Config", DataLoaderConfig)
            resolved.append(config_cls.model_validate(entry))
        self.loaders = resolved
        return self