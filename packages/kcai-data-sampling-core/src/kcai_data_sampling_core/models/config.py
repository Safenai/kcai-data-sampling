"""Root job configuration, and the registry-resolved transformation config.

The ``transformations`` entries of the YAML are validated against each
algorithm's own registered pydantic schema: core
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
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kcai_data_sampling_core.models.dataloaders import DataLoadersConfig
from kcai_data_sampling_core.models.global_ import ComputeConfig, ErrorsConfig, StorageConfig
from kcai_data_sampling_core.models.interfaces import SamplingInterfaceConfig
from kcai_data_sampling_core.models.models import ModelsConfig
from kcai_data_sampling_core.models.sweep import expand_sweeps


class ColumnsConfig(BaseModel):
    """Column routing on a transformation; generic, not per-algorithm.

    ``input`` names the source column(s) that feed the transformation.
    Reserved: a unary transformation reads the batch sample stack, so
    ``input`` is validated but not consumed — it is the n-ary seam, kept so
    n-ary transformations can widen it without a config-breaking change.

    Attributes:
        input: The input column name(s); when present must name the loader's
            ``sample_path.column``.
    """

    model_config = ConfigDict(extra="forbid")

    input: list[str] | None = Field(
        default=None,
        description="Input column names (reserved; must name the loader's sample column).",
    )


class TransformationConfig(BaseModel):
    """Base transformation configuration; algorithm packages subclass it.

    The base carries the config keys shared by every transformation and
    validates ``type`` against the loaded registry. Each algorithm package
    subclasses this with its own fields (and pins ``type`` to its literal),
    and registers the **algorithm class** that carries the subclass as its
    ``Config`` attribute — that is how config validation finds the specific
    schema.

    Attributes:
        name: Unique name of the transformation within the interface; optional
            (the schema doubles as the transformation instance's parameter
            validator, which has no name).
        type: Registry-resolved algorithm type (the transformation's identity).
        seed: Random seed; always accepted but only drawn when the algorithm
            is stochastic.
        storage: Optional storage override.
        columns: Reserved column routing (the n-ary seam); see
            :class:`ColumnsConfig`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        description="Unique name within the interface (optional on the instance schema).",
    )
    type: str = Field(description="Registry-resolved transformation type.")
    seed: int | None = Field(default=None, description="Random seed; drawn only when stochastic.")
    storage: bool | dict[str, Any] | None = Field(
        default=None,
        description="Optional storage override (local only).",
    )
    columns: ColumnsConfig | None = Field(
        default=None,
        description="Reserved column routing (the n-ary seam).",
    )

    @field_validator("type")
    @classmethod
    def _known(cls, v: str) -> str:
        """Refuse transformation types the registry does not know.

        Args:
            v: The ``type`` string from the config.

        Returns:
            The validated type string.

        Raises:
            ValueError: If the type is not registered.
        """
        from kcai_data_sampling_core.utils.registry import get_transformations_registry

        if v not in get_transformations_registry():
            raise ValueError(f"unknown transformation type {v!r}")
        return v


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

    @field_validator("operations")
    @classmethod
    def _resolve_operations(cls, v: SamplingInterfaceConfig) -> SamplingInterfaceConfig:
        """Resolve each raw transformation entry against its registered schema.

        The YAML entries are plain dicts; this validator looks each ``type`` up
        in the transformation registry, takes the algorithm's own config schema
        (``algorithm.Config``), and validates the entry against it, so required
        parameters and unknown fields are refused at config load.

        Args:
            v: The partially validated interface config.

        Returns:
            The interface config with every transformation validated against
            its specific registered schema.

        Raises:
            ValueError: If a transformation type is unknown or an entry does
                not pass its algorithm's schema.
        """
        from kcai_data_sampling_core.utils.registry import get_transformations_registry

        registry = get_transformations_registry()
        resolved: list[TransformationConfig] = []
        for entry in v.transformations:
            entry = dict(entry)
            algo = registry.get(entry.get("type", ""))
            if algo is None:
                raise ValueError(f"unknown transformation type {entry.get('type')!r}")
            config_cls = getattr(algo, "Config", TransformationConfig)
            resolved.append(config_cls.model_validate(entry))
        v.transformations = resolved
        return v

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
        used = [
            t for t in self.operations.transformations
            if t.columns is not None and t.columns.input is not None
        ]
        if not used:
            return self
        sample_columns = {
            sp.column
            for loader in self.dataloaders.loaders
            for sp in (loader.sample_path or [])
        }
        for entry in used:
            value = entry.columns.input
            if len(value) != 1:
                raise ValueError(
                    f"{entry.name}: columns.input must name exactly one column, got {value!r}"
                )
            if value[0] not in sample_columns:
                raise ValueError(
                    f"{entry.name}: columns.input {value[0]!r} is not any loader's "
                    f"sample_path.column (loaders: {sorted(sample_columns) or 'none declared'})"
                )
        return self
