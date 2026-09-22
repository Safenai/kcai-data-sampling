"""Base transformation class.

    T : x ↦ x′

A transformation is one fully specified operation: algorithm + resolved
parameters + seed + model. The family is derived from the role of the model in
producing the output; everything else is declared by the algorithm.
"""

import hashlib
from typing import Any

import numpy as np

from kcai_data_sampling_core.api.roles import check_model

#: Family = role of the model in computing the output. Never declared by an
#: algorithm: it is derived from ``model_role``.
FAMILY_BY_ROLE: dict[str | None, str] = {
    None: "procedural",
    "tool": "generative",
    "target": "adversarial",
}


class Transformation:
    """Shared identity and declarations for every transformation.

    Lifecycle methods live in the arity subclasses, :class:`UnaryTransformation`
    and (from phase 7) :class:`NAryTransformation`. Construction validates the
    resolved parameters against the algorithm's ``parameters`` declaration, and
    enforces that exactly the slot named by ``model_role`` is filled.

    Attributes:
        algorithm: Set by every algorithm; the registry-resolved ``type``.
        model_role: The model's role, ``None | "tool" | "target"``, which fixes
            the family and the slot to fill.
        model_methods: The methods the model must expose (see ``api.roles``),
            checked at construction.
        parameters: The algorithm's parameters: ``None`` for a required one,
            otherwise its default. ``params`` is then always complete.
        stochastic: Does the algorithm draw randomness? Only then does a seed
            draw.
        arity: ``"unary"`` (``"n-ary"`` from phase 7), fixed by the base class.
        clips: May the output leave the selection's value range? A declaration:
            the base clips to the range only when this is set; an undeclared
            overflow is refused.
        reversible: Is ``x`` determined by ``x′``? A structural declaration,
            frozen on the row for the downstream judge.
        seed: The random seed; always accepted by the config, but only drawn
            when ``stochastic`` (relaxed seed rule).
        params: The fully resolved parameters, complete.
        tool_model: The tool model instance, or ``None``.
        target_model: The target model instance, or ``None``.
    """

    #: Set by every algorithm.
    algorithm: str = "?"

    #: The model, if any: its role, None | "tool" | "target", which fixes the
    #: family and the slot to fill; and the methods that model must expose
    #: (see ``api.roles``), checked at construction.
    model_role: str | None = None
    model_methods: tuple[str, ...] = ()

    #: The algorithm's parameters: ``None`` for a required one, otherwise its
    #: default. Checked at construction; ``params`` is then always complete,
    #: so the row carries every parameter resolved.
    parameters: dict[str, Any] = {}

    #: Does the algorithm draw randomness? Only then does a seed exist.
    stochastic: bool = False

    #: "unary" | "n-ary", fixed by the base class.
    arity: str = "unary"

    #: Can the output leave the sample's range of values, so that it has to be
    #: clipped back to it? A declaration: the base clips, to the range the
    #: selection declares, only when it is set. An undeclared overflow is
    #: refused. The range itself has one source, the selection.
    clips: bool = False

    #: Is `x` determined by `x′`? A structural declaration about the map (a
    #: flip is a bijection, a crop discards), on trust: resolved parameters can
    #: break it where the range clips. Frozen on the row for the downstream
    #: judge; the interface itself draws nothing from it.
    reversible: bool = False

    def __init__(self, config: dict[str, Any] | None = None):
        """Build the transformation from a resolved configuration.

        Args:
            config: ``seed`` (accepted for any algorithm, drawn only when
                stochastic), ``tool_model``, ``target_model``, and any resolved
                algorithm parameter.

        Raises:
            ValueError: If a parameter is unknown or a required one is missing,
                if a seed is meaningless for a deterministic algorithm, or if
                the slot named by ``model_role`` is not filled exactly.
        """
        config = dict(config or {})
        seed = config.pop("seed", None)
        self.tool_model: Any = config.pop("tool_model", None)
        self.target_model: Any = config.pop("target_model", None)
        self.params: dict[str, Any] = self.resolve(config)

        # Relaxed seed rule: a seed is always accepted and recorded, but a
        # deterministic algorithm has no random draw, so it is stored as None.
        self.seed: int | None = 0 if seed is None else int(seed) if self.stochastic else None

        # Exactly the slot named by the role must be filled.
        needs = {"tool": self.tool_model, "target": self.target_model}
        for role, model in needs.items():
            if (model is not None) != (self.model_role == role):
                raise ValueError(
                    f"{self.algorithm}: model_role={self.model_role!r} but {role}_model="
                    f"{getattr(model, 'name', model)!r}, the family is the role of the model"
                )
        if self.model_role is not None:
            check_model(self.algorithm, self.model_role, needs[self.model_role], self.model_methods)

    def fit_to_range(self, out: np.ndarray, value_range: tuple[float, float] | None) -> np.ndarray:
        """Clip if the algorithm declared it; then check.

        Args:
            out: The algorithm's output batch.
            value_range: ``(low, high)`` domain of the selection, or ``None``
                to leave the output unchecked.

        Returns:
            The output, clipped when the algorithm declares ``clips``.

        Raises:
            ValueError: If the output leaves the range and the algorithm did
                not declare ``clips``.
        """
        if value_range is None:
            return out
        low, high = value_range
        if self.clips:
            return np.clip(out, low, high)
        lo, hi = float(out.min()), float(out.max())
        if lo < low or hi > high:
            raise ValueError(
                f"{self.algorithm} left the selection's value range [{low}, {high}]: output in "
                f"[{lo:.4g}, {hi:.4g}]. An algorithm whose output can leave it declares `clips = True`."
            )
        return out

    def resolve(self, given: dict[str, Any]) -> dict[str, Any]:
        """Complete the parameters: fill defaults, refuse unknowns.

        Args:
            given: The raw parameters from the config.

        Returns:
            The parameters, complete, with every declared default filled in.

        Raises:
            ValueError: If an unknown parameter is given or a required one is
                missing.
        """
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
        """The derived family: the role of the model in producing the output."""
        return FAMILY_BY_ROLE[self.model_role]

    def rngs(self, keys: list[str]) -> list[np.random.Generator] | None:
        """Yield one generator per output, seeded by the seed and the parent id(s).

        Seeding from the output's parent ids makes a draw independent of both
        the batch and the order of the run.

        Args:
            keys: One key per output (the parent id, or the joined parent ids
                for n-ary), used to derive each generator's seed.

        Returns:
            One ``np.random.Generator`` per key, or ``None`` when the algorithm
            is deterministic.
        """
        if not self.stochastic:
            return None
        return [np.random.default_rng([self.seed, int.from_bytes(hashlib.sha1(k.encode()).digest()[:8], "big")]) for k in keys]

    def describe(self) -> dict[str, Any]:
        """Return the declaration-level fields every output row carries.

        Returns:
            A dictionary with ``algorithm``, ``family``, ``arity``,
            ``reversible``, ``params``, ``seed``, ``tool_model`` and
            ``target_model`` (models lowered to their ``name``).
        """
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