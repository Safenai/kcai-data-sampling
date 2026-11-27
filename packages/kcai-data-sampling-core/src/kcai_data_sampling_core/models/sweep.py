"""Transformation ranges: one config parameter expanded into many values.

A ``SweepConfig`` widens an algorithm parameter into a sweep: ``mode: even``
samples ``numpy.linspace`` across the interval (inclusive ends), ``mode:
random`` draws seeded uniform values, ``step`` walks an arithmetic
progression, and ``values`` lists the values explicitly. Values are resolved
**once at config expansion** from the global ``compute.seed``, so a run stays
reproducible and batch-invariant.

The expansion helper (:func:`expand_sweeps`) turns a validated transformation
model with any ``SweepConfig``-valued fields into as many concrete models as
the cartesian product of the swept parameters; each concrete model carries the
expanded value in the swept field and an auto-suffixed ``name``; the derived
``id`` stays deterministic — it hashes resolved ``params``.
"""

import itertools
import typing
from typing import Any, Literal, Protocol, Self, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Expandable(Protocol):
    """The pydantic surface ``expand_sweeps`` relies on.

    Kept local so the expansion stays generic (any model carrying
    ``SweepConfig``-valued fields) without importing the transformation
    config and creating a models-cycle.
    """

    name: str | None

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> BaseModel: ...


TModel = TypeVar("TModel", bound=_Expandable)


class SweepConfig(BaseModel):
    """Expand one parameter into a range of concrete values.

    Exactly one of ``samples``, ``step`` and ``values`` is required.
    ``samples`` and ``step`` cover ``range``; ``values`` lists the values
    themselves and takes no ``range``. ``mode`` applies with ``samples`` only.

    Attributes:
        range: The inclusive ``[min, max]`` interval to cover, with
            ``samples`` or ``step``.
        samples: Number of values when sweeping by count; exclusive with
            ``step``.
        step: Step of the arithmetic progression; exclusive with ``samples``.
        mode: ``even`` (``linspace``) or ``random`` (seeded draws); applies
            with ``samples``.
        listed: The values themselves, in the order given (YAML key
            ``values``); exclusive with ``range``, ``samples`` and ``step``.
        round: Decimals kept on the expanded values; ``None`` (the default)
            keeps them as computed, up to 12 significant digits.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    range: list[float] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="The inclusive [min, max] interval to cover, with 'samples' or 'step'.",
    )
    samples: int | None = Field(
        default=None,
        gt=0,
        description="Number of values when sweeping by count; exclusive with 'step'.",
    )
    step: float | int | None = Field(
        default=None,
        gt=0,
        description="Step of the arithmetic progression; exclusive with 'samples'.",
    )
    mode: Literal["even", "random"] = Field(default="even", description="'even' (linspace) or 'random' (seeded draws).")
    listed: list[float] | None = Field(
        default=None,
        alias="values",
        min_length=1,
        description="The values themselves, in the order given; exclusive with 'range', 'samples' and 'step'.",
    )
    round: int | None = Field(
        default=None,
        ge=0,
        description="Decimals kept on the expanded values; by default none: the values are kept as computed.",
    )

    @field_validator("range")
    @classmethod
    def _ordered(cls, v: list[float] | None) -> list[float] | None:
        """Refuse inverted intervals (they would expand to nothing useful).

        Args:
            v: The ``[min, max]`` interval, or ``None``.

        Returns:
            The interval as given, when ``min <= max``.

        Raises:
            ValueError: If ``min > max``.
        """
        if v is not None and v[0] > v[1]:
            raise ValueError(f"sweep range {v} must satisfy min <= max")
        return v

    @model_validator(mode="after")
    def _exclusive(self) -> Self:
        """Enforce exactly one of ``samples``, ``step`` and ``values``.

        Returns:
            The validated sweep.

        Raises:
            ValueError: If not exactly one of them is set, if ``range`` is
                missing with ``samples`` or ``step`` or given with
                ``values``, or if ``values`` repeats a value.
        """
        if sum(x is not None for x in (self.samples, self.step, self.listed)) != 1:
            raise ValueError("a sweep needs exactly one of 'samples', 'step' or 'values'")
        if self.listed is not None:
            if self.range is not None:
                raise ValueError("a sweep with 'values' takes no 'range'")
            if len(set(self.listed)) != len(self.listed):
                raise ValueError(f"sweep values {self.listed} repeat a value")
        elif self.range is None:
            raise ValueError("a sweep with 'samples' or 'step' needs a 'range'")
        return self

    @property
    def bounds(self) -> tuple[float, float]:
        """The inclusive ``(min, max)`` the sweep covers: its range, or its smallest and largest values."""
        if self.listed is not None:
            return (min(self.listed), max(self.listed))
        assert self.range is not None
        return (self.range[0], self.range[1])

    @property
    def shown(self) -> list[float]:
        """What the sweep was given, for messages: its ``range``, or its ``values``."""
        return list(self.listed if self.listed is not None else self.range or [])

    def values(self, seed: int | None = None) -> list[float]:
        """Expand this sweep to its concrete values.

        ``even`` uses ``linspace(min, max, samples)`` (inclusive ends), ``step``
        walks ``min, min+step, ...`` up to ``max`` included (a final partial
        step is dropped), ``random`` draws ``samples`` uniform values in
        ``[min, max]`` from the seed, and ``values`` gives them as listed. The
        values are kept as computed, to 12 significant digits (which
        drops the float noise of ``linspace`` and of the step walk, such as
        ``0.15000000000000002``, without collapsing small values), or rounded
        to ``round`` decimals when it is set.

        Args:
            seed: Seed for the ``random`` draws; required by that mode,
                ignored otherwise.

        Returns:
            The expanded values, in ascending order (draw order for
            ``random``, listed order for ``values``).

        Raises:
            ValueError: If ``mode: random`` is used without a seed (it would
                otherwise be non-deterministic).
        """
        if self.listed is not None:
            raw = list(self.listed)
            return self._finish(raw)
        assert self.range is not None
        lo, hi = self.range
        if self.step is not None:
            step = float(self.step)
            # Count the steps on the cleaned quotient: 0.3 / 0.1 is 2.9999999999999996, which must give 3.
            count = int(_clean((hi - lo) / step))
            raw = [lo + i * step for i in range(count + 1)]
            raw = [v for v in raw if _clean(v) <= hi]
        elif self.mode == "even":
            assert self.samples is not None
            raw = np.linspace(lo, hi, self.samples).tolist()
        else:
            if seed is None:
                raise ValueError("mode: random needs the compute seed to stay deterministic")
            assert self.samples is not None
            rng = np.random.default_rng(seed)
            raw = rng.uniform(lo, hi, size=self.samples).tolist()
        return self._finish(raw)

    def _finish(self, raw: list[float]) -> list[float]:
        """Clean the float noise off the values, or round them to ``round`` decimals when it is set.

        Args:
            raw: The expanded values.

        Returns:
            The values to use.
        """
        if self.round is None:
            return [_clean(v) for v in raw]
        return [round(v, self.round) for v in raw]


def _clean(value: float) -> float:
    """Keep 12 significant digits: drops float noise (``0.15000000000000002``), never collapses small values.

    Args:
        value: A computed value.

    Returns:
        The value without its float noise.
    """
    return float(f"{value:.12g}")


def _coerce_to_field(value: float, annotation: Any) -> Any:
    """Cast an expanded value to the swept field's type.

    A union like ``float | SweepConfig`` that also accepts ``int`` receives an
    ``int`` (integer params swept via ``step``); float params keep their float.

    Args:
        value: The expanded value.
        annotation: The swept field's type annotation.

    Returns:
        The value cast to ``int`` when the field accepts it, else as-is.
    """
    args = typing.get_args(annotation)
    if int in args:
        return int(value)
    return value


def _resolve_annotation(model: Any, name: str) -> Any:
    """Return the declared type of a config model field.

    Args:
        model: The validated config model instance.
        name: The field name.

    Returns:
        The field's declared annotation.
    """
    return model.__class__.model_fields[name].annotation


def expand_sweeps(model: TModel, seed: int | None = None) -> list[TModel]:
    """Expand a validated model's ``SweepConfig`` fields into concrete values.

    Args:
        model: The validated transformation config.
        seed: Global ``compute.seed``, handed to the ``random`` draws.

    Returns:
        The model itself when it holds no sweep, else the concrete models of
        the cartesian product of the swept parameters. Each model has the
        swept field replaced by one expanded value, type-cast to the field,
        and ``name`` auto-suffixed (``crop_grid__0.27``; several swept params
        joined by ``_``).
    """
    swept = {name: value for name, value in model.__dict__.items() if isinstance(value, SweepConfig)}
    if not swept:
        return [model]

    options = {name: sweep.values(seed) for name, sweep in swept.items()}
    variants: list[TModel] = []
    for combo in itertools.product(*(options[name] for name in options)):
        updates = {
            name: _coerce_to_field(value, _resolve_annotation(model, name))
            for name, value in zip(options, combo, strict=True)
        }
        updates["name"] = f"{model.name}__" + "_".join(f"{updates[name]}" for name in options)
        variants.append(typing.cast(TModel, model.model_copy(update=updates)))
    return variants
