"""Transformation ranges: one config parameter expanded into many values.

A ``SweepConfig`` widens an algorithm parameter into a sweep: ``mode: even``
samples ``numpy.linspace`` across the interval (inclusive ends), ``mode:
random`` draws seeded uniform values, and ``step`` walks an arithmetic
progression. Values are resolved **once at config expansion** from the global
``compute.seed``, so a run stays reproducible and batch-invariant.

The expansion helper (:func:`expand_sweeps`) turns a validated transformation
model with any ``SweepConfig``-valued fields into as many concrete models as
the cartesian product of the swept parameters; each concrete model carries the
rounded value in the swept field and an auto-suffixed ``name``; the derived
``id`` stays deterministic — it hashes resolved ``params``.
"""

import itertools
import typing
from typing import Any, Protocol, Self, TypeVar

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

    ``samples`` and ``step`` are mutually exclusive; exactly one is required.
    ``mode`` applies with ``samples`` only.

    Attributes:
        range: The inclusive ``[min, max]`` interval to cover.
        samples: Number of values when sweeping by count; exclusive with
            ``step``.
        step: Step of the arithmetic progression; exclusive with ``samples``.
        mode: ``even`` (``linspace``) or ``random`` (seeded draws); applies
            with ``samples``.
        round: Decimals kept on the expanded values (default 2).
    """

    model_config = ConfigDict(extra="forbid")

    range: list[float] = Field(
        min_length=2,
        max_length=2,
        description="The inclusive [min, max] interval to cover.",
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
    mode: str = Field(default="even", description="'even' (linspace) or 'random' (seeded draws).")
    round: int = Field(
        default=2,
        ge=0,
        description="Decimals kept on the expanded values (default 2).",
    )

    @field_validator("range")
    @classmethod
    def _ordered(cls, v: list[float]) -> list[float]:
        """Refuse inverted intervals (they would expand to nothing useful).

        Args:
            v: The ``[min, max]`` interval.

        Returns:
            The interval as given, when ``min <= max``.

        Raises:
            ValueError: If ``min > max``.
        """
        if v[0] > v[1]:
            raise ValueError(f"sweep range {v} must satisfy min <= max")
        return v

    @model_validator(mode="after")
    def _exclusive(self) -> Self:
        """Enforce exactly one of ``samples`` or ``step``.

        Returns:
            The validated sweep.

        Raises:
            ValueError: If both ``samples`` and ``step`` are set, or neither
                is.
        """
        if (self.samples is None) == (self.step is None):
            raise ValueError("a sweep needs exactly one of 'samples' or 'step'")
        return self

    @property
    def bounds(self) -> tuple[float, float]:
        """The inclusive ``(min, max)`` interval as a tuple."""
        return (self.range[0], self.range[1])

    def values(self, seed: int | None = None) -> list[float]:
        """Expand this sweep to its concrete values, each rounded to ``round``.

        ``even`` uses ``linspace(min, max, samples)`` (inclusive ends), ``step``
        walks ``min, min+step, ...`` (a final partial step is dropped), and
        ``random`` draws ``samples`` uniform values in ``[min, max]`` from the
        seed.

        Args:
            seed: Seed for the ``random`` draws; required by that mode,
                ignored otherwise.

        Returns:
            The expanded values, in ascending order (draw order for
            ``random``) and rounded to ``round`` decimals.

        Raises:
            ValueError: If ``mode: random`` is used without a seed (it would
                otherwise be non-deterministic).
        """
        lo, hi = self.range
        if self.step is not None:
            step = float(self.step)
            raw = [lo + i * step for i in range(int((hi - lo) / step) + 1)]
            raw = [v for v in raw if v <= hi + 1e-9]
        elif self.mode == "even":
            assert self.samples is not None
            raw = np.linspace(lo, hi, self.samples).tolist()
        else:
            if seed is None:
                raise ValueError("mode: random needs the compute seed to stay deterministic")
            assert self.samples is not None
            rng = np.random.default_rng(seed)
            raw = rng.uniform(lo, hi, size=self.samples).tolist()
        return [round(v, self.round) for v in raw]


def _coerce_to_field(value: float, annotation: Any) -> Any:
    """Cast an expanded value to the swept field's type.

    A union like ``float | SweepConfig`` that also accepts ``int`` receives an
    ``int`` (integer params swept via ``step``); float params keep their float.

    Args:
        value: The rounded expanded value.
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
