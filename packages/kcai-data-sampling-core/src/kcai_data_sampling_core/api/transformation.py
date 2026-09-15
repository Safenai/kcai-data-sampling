"""Base transformation class.

    T : x ↦ x′

A transformation is one fully specified operation: algorithm + resolved
parameters + seed + model. The family is derived from the model's role; the
rest is declared by the algorithm.
"""

import hashlib
from typing import Any

import numpy as np

from kcai_data_sampling_core.api.roles import check_model

#: Family = role of the model in computing the output.
FAMILY_BY_ROLE: dict[str | None, str] = {
    None: "procedural",
    "tool": "generative",
    "target": "adversarial",
}


class Transformation:
    """Shared identity and declarations. Lifecycle methods live in the
    arity subclasses, :class:`UnaryTransformation` and :class:`NAryTransformation`."""

    #: Set by every algorithm.
    algorithm: str = "?"

    #: None | "tool" | "target", fixes the family and which model slot is required.
    model_role: str | None = None

    #: The methods the model in that slot must expose (see ``api.roles``).
    requires: tuple[str, ...] = ()

    #: The algorithm's parameters: ``None`` for a required one, otherwise its
    #: default. Checked at construction; ``params`` is then always complete,
    #: so the row carries every parameter resolved.
    parameters: dict[str, Any] = {}

    #: Does the algorithm draw randomness? Only then does a seed exist.
    stochastic: bool = False

    #: "unary" | "n-ary", fixed by the base class.
    arity: str = "unary"

    #: Is `x` determined by `x′`? A structural declaration about the map (a
    #: flip is a bijection, a crop discards), on trust: resolved parameters
    #: can break it where the range clips. Frozen on the row for the
    #: downstream judge; the interface itself draws nothing from it.
    reversible: bool = False

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Args:
            config: `seed` (only if stochastic), `tool_model`, `target_model`,
                and any resolved algorithm parameter.
        """
        config = dict(config or {})
        seed = config.pop("seed", None)
        self.tool_model: Any = config.pop("tool_model", None)
        self.target_model: Any = config.pop("target_model", None)
        self.params: dict[str, Any] = self.resolve(config)

        if self.stochastic:
            self.seed: int | None = 0 if seed is None else int(seed)
        elif seed is not None:
            raise ValueError(
                f"{self.algorithm} draws no randomness: a seed means nothing here, "
                "and two seeds would name two transformations for one output"
            )
        else:
            self.seed = None

        # Exactly the slot named by the role must be filled.
        needs = {"tool": self.tool_model, "target": self.target_model}
        for role, model in needs.items():
            if (model is not None) != (self.model_role == role):
                raise ValueError(
                    f"{self.algorithm}: model_role={self.model_role!r} but {role}_model="
                    f"{getattr(model, 'name', model)!r}, the family is the role of the model"
                )
        if self.model_role is not None:
            check_model(self.algorithm, self.model_role, needs[self.model_role], self.requires)

    def resolve(self, given: dict[str, Any]) -> dict[str, Any]:
        """The parameters, complete: defaults filled in, a missing required one
        or an unknown key refused."""
        unknown = sorted(set(given) - set(self.parameters))
        if unknown:
            raise ValueError(
                f"{self.algorithm} does not take {', '.join(map(repr, unknown))}; "
                f"it takes {', '.join(self.parameters) or 'no parameter'}"
            )
        missing = [k for k, d in self.parameters.items() if d is None and k not in given]
        if missing:
            optional = ", ".join(f"{k} ({d!r})" for k, d in self.parameters.items() if d is not None)
            raise ValueError(f"{self.algorithm} needs {', '.join(missing)}" + (f"; optional: {optional}" if optional else ""))
        return {k: given.get(k, d) for k, d in self.parameters.items()}

    @property
    def family(self) -> str:
        return FAMILY_BY_ROLE[self.model_role]

    def rngs(self, keys: list[str]) -> list[np.random.Generator] | None:
        """One generator per output, seeded by the seed and the output's parent
        id(s), so that a draw depends on neither the batch nor the order of the
        run. `None` when deterministic."""
        if not self.stochastic:
            return None
        return [np.random.default_rng([self.seed, int.from_bytes(hashlib.sha1(k.encode()).digest()[:8], "big")]) for k in keys]

    def describe(self) -> dict[str, Any]:
        """The declaration-level fields every output row carries."""
        return {
            "algorithm": self.algorithm,
            "family": self.family,
            "arity": self.arity,
            "reversible": self.reversible,
            "params": dict(self.params),
            "seed": self.seed,
            "tool_model": getattr(self.tool_model, "name", None),
            "target_model": getattr(self.target_model, "name", None),
        }
