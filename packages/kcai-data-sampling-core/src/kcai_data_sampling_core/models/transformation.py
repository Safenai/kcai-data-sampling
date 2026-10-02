"""The registry-resolved transformation configuration.

The ``transformations`` entries of the YAML are validated against each
algorithm's own registered pydantic schema: core cannot enumerate the
discriminated union because algorithm packages arrive after core, so the
``type`` string is looked up in the loaded transformation registry and the
entry is validated against the registered schema. Unknown or mistyped types
are refused at config load.

The base lives here (not in ``config.py``) so the interface model can type its
``transformations`` field against it without an import cycle.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kcai_data_sampling_core.models.sweep import SweepConfig


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

    @staticmethod
    def _at_or_below(value: float, bound: float | None, exclusive: bool) -> bool:
        """Whether a value crosses a lower side (inclusive unless ``exclusive``).

        Args:
            value: The value or sweep-interval end to check.
            bound: The lower bound to enforce, or ``None`` for an open side.
            exclusive: Whether the bound is exclusive.

        Returns:
            ``True`` when ``bound`` is set and ``value`` falls below it.
        """
        return bound is not None and (value < bound or (exclusive and value == bound))

    @staticmethod
    def _at_or_above(value: float, bound: float | None, exclusive: bool) -> bool:
        """Whether a value crosses an upper side (inclusive unless ``exclusive``).

        Args:
            value: The value or sweep-interval end to check.
            bound: The upper bound to enforce, or ``None`` for an open side.
            exclusive: Whether the bound is exclusive.

        Returns:
            ``True`` when ``bound`` is set and ``value`` exceeds it.
        """
        return bound is not None and (value > bound or (exclusive and value == bound))

    def _check_parameter_bounds(
        self,
        name: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_min: bool = False,
        exclusive_max: bool = False,
    ) -> None:
        """Refuse a parameter value or sweep interval outside its range.

        The shared bound check for parameterized algorithms: each subclass
        calls it per parameter, and the sweep-or-value dispatch, the interval
        rule and the message wording live here once instead of being
        reimplemented in every algorithm config. The ``<type>`` prefix is the
        subclass's pinned literal, so the messages read exactly as before.

        Args:
            name: The parameter name, for the error message.
            minimum: The smallest allowed value (inclusive unless
                ``exclusive_min``); ``None`` leaves the lower side open.
            maximum: The largest allowed value (inclusive unless
                ``exclusive_max``); ``None`` leaves the upper side open.
            exclusive_min: Whether the lower bound is exclusive.
            exclusive_max: Whether the upper bound is exclusive.

        Raises:
            ValueError: If the value — or, for a sweep, its interval — leaves
                the range.
        """
        value = getattr(self, name)
        if isinstance(value, SweepConfig):
            lo, hi = value.bounds
            if self._at_or_below(lo, minimum, exclusive_min):
                raise ValueError(
                    self._sweep_bounds_error(name, value.range, minimum, maximum, exclusive_min, exclusive_max)
                )
            if self._at_or_above(hi, maximum, exclusive_max):
                raise ValueError(
                    self._sweep_bounds_error(name, value.range, minimum, maximum, exclusive_min, exclusive_max)
                )
            return
        if self._at_or_below(value, minimum, exclusive_min):
            raise ValueError(self._bounds_error(name, value, minimum, maximum, exclusive_min, exclusive_max))
        if self._at_or_above(value, maximum, exclusive_max):
            raise ValueError(self._bounds_error(name, value, minimum, maximum, exclusive_min, exclusive_max))

    def _bounds_error(
        self,
        name: str,
        shown: Any,
        minimum: float | None,
        maximum: float | None,
        exclusive_min: bool,
        exclusive_max: bool,
    ) -> str:
        """Format a plain-value bound violation message.

        Args:
            name: The parameter name.
            shown: The offending value.
            minimum: The lower bound, or ``None``.
            maximum: The upper bound, or ``None``.
            exclusive_min: Whether the lower bound is exclusive.
            exclusive_max: Whether the upper bound is exclusive.

        Returns:
            The message: ``<type>.<name> must be in (min, max], got ...`` for
            a two-sided bound, else ``must be >`` / ``>= `` / ``<= `` single
            bounds.
        """
        if minimum is not None and maximum is not None:
            lo = "(" if exclusive_min else "["
            hi = ")" if exclusive_max else "]"
            return f"{self.type}.{name} must be in {lo}{minimum}, {maximum}{hi}, got {shown}"
        if minimum is not None:
            comparator = ">" if exclusive_min else ">="
            return f"{self.type}.{name} must be {comparator} {minimum}, got {shown}"
        comparator = "<" if exclusive_max else "<="
        return f"{self.type}.{name} must be {comparator} {maximum}, got {shown}"

    def _sweep_bounds_error(
        self,
        name: str,
        range_: list[float],
        minimum: float | None,
        maximum: float | None,
        exclusive_min: bool,
        exclusive_max: bool,
    ) -> str:
        """Format a sweep-interval bound violation message.

        Args:
            name: The parameter name.
            range_: The offending interval.
            minimum: The lower bound, or ``None``.
            maximum: The upper bound, or ``None``.
            exclusive_min: Whether the lower bound is exclusive.
            exclusive_max: Whether the upper bound is exclusive.

        Returns:
            The message: ``<type>.<name> sweep <range> must stay within (min,
            max]`` for a two-sided bound, else ``must stay >`` / ``>=`` /
            ``<=`` single bounds.
        """
        if minimum is not None and maximum is not None:
            lo = "(" if exclusive_min else "["
            hi = ")" if exclusive_max else "]"
            return f"{self.type}.{name} sweep {range_} must stay within {lo}{minimum}, {maximum}{hi}"
        if minimum is not None:
            comparator = ">" if exclusive_min else ">="
            return f"{self.type}.{name} sweep {range_} must stay {comparator} {minimum}"
        comparator = "<" if exclusive_max else "<="
        return f"{self.type}.{name} sweep {range_} must stay {comparator} {maximum}"
