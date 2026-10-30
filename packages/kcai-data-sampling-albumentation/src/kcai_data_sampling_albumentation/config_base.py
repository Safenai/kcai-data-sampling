"""Shared config base for every generated albu-backed transformation schema.

Hand-written (not generated): the generated ``configs.py`` subclasses
:class:`AlbumentationsConfig` and pins ``type`` plus one field per exposed
parameter. Carries the three shared validators — probability bounds, the seed
requirement, and the range-pair-or-scalar-or-sweep bounds check, which the core
``_check_parameter_bounds`` cannot express because a pair is a two-element
list.
"""

from typing import Self

from kcai_data_sampling_core.models.config import TransformationConfig
from kcai_data_sampling_core.models.sweep import SweepConfig
from pydantic import Field


class AlbumentationsConfig(TransformationConfig):
    """Base schema of the albumentations-backed transformations.

    Attributes:
        p: Application probability, default 1.0 (decided: a sampling entry
            names an experiment; upstream's 0.5 would silently emit identity
            outputs mislabeled as transformed). A sweep fixes it per variant.
    """

    p: float | SweepConfig = Field(
        default=1.0,
        description="Application probability; a SweepConfig expands it. Sampling default is 1.0 (always apply).",
    )

    def _check_range_pair(
        self,
        name: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_min: bool = False,
        exclusive_max: bool = False,
    ) -> None:
        """Refuse a pair/scalar/sweep parameter outside its algorithm range.

        A ``[lo, hi]`` pair (per-row draw range) is checked element-wise and
        for ordering; a scalar (fixed magnitude) and a sweep interval go
        through the core ``_check_parameter_bounds``.

        Args:
            name: The parameter name, for the error message.
            minimum: Smallest allowed value (inclusive unless
                ``exclusive_min``); ``None`` leaves the lower side open.
            maximum: Largest allowed value (inclusive unless
                ``exclusive_max``); ``None`` leaves the upper side open.
            exclusive_min: Whether the lower bound is exclusive.
            exclusive_max: Whether the upper bound is exclusive.

        Raises:
            ValueError: If a pair element, the scalar, or the sweep interval
                leaves the range, or the pair is unordered.
        """
        value = getattr(self, name)
        if not isinstance(value, list):
            self._check_parameter_bounds(
                name, minimum=minimum, maximum=maximum, exclusive_min=exclusive_min, exclusive_max=exclusive_max
            )
            return
        if len(value) != 2:
            raise ValueError(f"{self.type}.{name} must be a [lo, hi] pair, got {value}")
        if value[0] > value[1]:
            raise ValueError(f"{self.type}.{name} must satisfy lo <= hi, got {value}")
        for element in value:
            if self._at_or_below(element, minimum, exclusive_min):
                raise ValueError(self._bounds_error(name, element, minimum, maximum, exclusive_min, exclusive_max))
            if self._at_or_above(element, maximum, exclusive_max):
                raise ValueError(self._bounds_error(name, element, minimum, maximum, exclusive_min, exclusive_max))

    def _check_p(self) -> None:
        """Enforce ``p`` in ``[0, 1]``, value or sweep interval.

        Raises:
            ValueError: If ``p`` leaves ``[0, 1]``.
        """
        self._check_parameter_bounds("p", minimum=0, maximum=1)

    def _require_seed(self) -> Self:
        """Refuse a config with no seed (every albu algorithm is stochastic).

        Returns:
            The validated config.

        Raises:
            ValueError: If ``seed`` is not set.
        """
        if self.seed is None:
            raise ValueError(
                f"{self.type}: a seed is required (every albumentations-backed transformation is stochastic)"
            )
        return self
