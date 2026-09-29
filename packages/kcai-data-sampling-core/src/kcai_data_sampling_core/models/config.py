"""Root job configuration.

The input ``transformations`` entries are validated against each algorithm's
own registered pydantic schema inside :class:`SamplingInterfaceConfig`: core
cannot enumerate the discriminated union because algorithm packages arrive
after core, so the ``type`` string is looked up in the loaded transformation
registry and the entry is validated against the registered schema. Unknown or
mistyped types are refused at config load.

A validated entry whose parameters carry ``SweepConfig`` values is *expanded*
at validation, before the registry consumers see it: each swept parameter
becomes as many concrete models as the values it covers (cartesian product
across several swept params), so the transformations list the job sees is
plain and long, and the id/params/artifact machinery is inherited
untouched.

``ColumnsConfig`` and ``TransformationConfig`` live in
:mod:`kcai_data_sampling_core.models.transformation`; they are re-imported
here so ``from kcai_data_sampling_core.models.config import TransformationConfig``
keeps working for algorithm packages and tests.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kcai_data_sampling_core.models.dataloaders import DataLoadersConfig
from kcai_data_sampling_core.models.global_ import ComputeConfig, ErrorsConfig, StorageConfig
from kcai_data_sampling_core.models.interfaces import SamplingInterfaceConfig
from kcai_data_sampling_core.models.models import ModelsConfig
from kcai_data_sampling_core.models.sweep import expand_sweeps
from kcai_data_sampling_core.models.transformation import ColumnsConfig as ColumnsConfig
from kcai_data_sampling_core.models.transformation import TransformationConfig as TransformationConfig


class JobConfig(BaseModel):
    """Root configuration for a sample-generation job. Each field maps to a pipeline stage.

    Attributes:
        storage: Global storage config (local only).
        compute: Global compute / runtime settings.
        errors: Global error-handling policy.
        dataloaders: The data loaders and their selections.
        models: Named tool / target model references.
        operations: The single transform stage: the list of transformations
            (each applied independently to the same selection), outputs, and
            the batch/error/storage overrides.
    """

    model_config = ConfigDict(extra="forbid")

    storage: StorageConfig | None = None
    compute: ComputeConfig | None = None
    errors: ErrorsConfig | None = None
    dataloaders: DataLoadersConfig = Field(description="Data loaders.")
    models: ModelsConfig | None = None
    operations: SamplingInterfaceConfig = Field(description="The single transformations interface.")

    @model_validator(mode="after")
    def _expand_sweeps(self) -> Self:
        """Expand any ``SweepConfig``-valued parameters at config validation.

        The sweep expansion happens here, once the ``operations`` and
        ``compute`` fields are validated: each swept transformation is replaced
        by as many concrete models as its values cover (cartesian product when
        several parameters sweep), seeded from ``compute.seed`` so ``mode:
        random`` draws are reproducible. The sweep has no seed of its own.

        Returns:
            The complete job config with sweeps expanded.

        Raises:
            ValueError: If a sweep is void or ``mode: random`` has no seed.
        """
        seed = self.compute.seed if self.compute is not None else ComputeConfig().seed
        expanded: list[TransformationConfig] = []
        for entry in self.operations.transformations:
            expanded.extend(expand_sweeps(entry, seed))
        self.operations.transformations = expanded
        return self

    @model_validator(mode="after")
    def _check_columns_input(self) -> Self:
        """Refuse a reserved ``columns.input`` that does not name a sample column.

        ``columns.input`` is optional and, when present, must be a
        list of exactly one value — a loader's ``sample_path.column`` (the
        single processable sample column each row carries at load). The value
        is not consumed; the transformation reads the batch sample
        stack as always. The check runs after sweeps expanded, so every
        concrete variant is covered.

        Returns:
            The validated job config.

        Raises:
            ValueError: If ``columns.input`` names more than one column, or
                names a column no loader declares as its ``sample_path.column``.
        """
        used = [t for t in self.operations.transformations if t.columns is not None and t.columns.input is not None]
        if not used:
            return self
        sample_columns = {sp.column for loader in self.dataloaders.loaders for sp in (loader.sample_path or [])}
        for entry in used:
            columns = entry.columns
            assert columns is not None
            input_columns = columns.input
            assert input_columns is not None
            if len(input_columns) != 1:
                raise ValueError(f"{entry.name}: columns.input must name exactly one column, got {input_columns!r}")
            if input_columns[0] not in sample_columns:
                raise ValueError(
                    f"{entry.name}: columns.input {input_columns[0]!r} is not any loader's "
                    f"sample_path.column (loaders: {sorted(sample_columns) or 'none declared'})"
                )
        return self
