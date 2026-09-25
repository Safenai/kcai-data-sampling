"""Base transformation class.

    T : x ↦ x′

A transformation is one fully specified operation: algorithm + resolved
parameters + seed + model. The family describes what the map does — declared
by the algorithm, defaulting to the role of the model in producing the output
when it is not declared; everything else is declared by the algorithm.
"""

import hashlib
import json
from typing import Any

import numpy as np

from kcai_data_sampling_core.api.roles import check_model

#: Default family = role of the model in producing the output. Used only when
#: an algorithm does not declare ``family`` itself (role says where the model
#: plugs in; family says what the map does).
FAMILY_BY_ROLE: dict[str | None, str] = {
    None: "procedural",
    "tool": "generative",
    "target": "adversarial",
}


class Transformation:
    """Shared identity and declarations for every transformation.

    Lifecycle methods live in the arity subclasses, :class:`UnaryTransformation`
    and, later, :class:`NAryTransformation`. Construction resolves the
    parameters **through the algorithm's registered pydantic ``Config``** — the
    single declaration of parameters and defaults, so a bad config fails fast
    at load — and enforces that exactly the slot named by ``model_role`` is
    filled.

    Attributes:
        algorithm: Set by every algorithm; the registry-resolved ``type``.
        family: The map's kind, ``None`` to derive it from the role of the
            model: ``"procedural"`` (no model), ``"generative"`` (tool) or
            ``"adversarial"`` (target).
        Config: The registered pydantic schema the algorithm's params are
            validated against; declared by every algorithm package.
        model_role: The model's role, ``None | "tool" | "target"``, which fixes
            the slot to fill.
        model_methods: The methods the model must expose (see ``api.roles``),
            checked at construction.
        stochastic: Does the algorithm draw randomness? Only then does a seed
            draw.
        arity: ``"unary"`` (``"n-ary"`` later), fixed by the base class.
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

    #: The map's kind, or ``None`` to derive it from the model's role.
    family: str | None = None

    #: The registered pydantic schema of this algorithm's parameters;
    #: validated at construction (``model_validate`` → dump). An algorithm
    #: without one declares no parameters.
    Config: Any = None

    #: The model, if any: its role, None | "tool" | "target", which fixes the
    #: slot to fill; and the methods that model must expose (see ``api.roles``),
    #: checked at construction.
    model_role: str | None = None
    model_methods: tuple[str, ...] = ()

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
            ValueError: If a parameter fails the registered ``Config`` schema
                (unknown or missing), if the slot named by ``model_role``
                is not filled exactly, or if a ``stochastic`` algorithm is
                given no seed.
        """
        config = dict(config or {})
        seed = config.pop("seed", None)
        self.tool_model: Any = config.pop("tool_model", None)
        self.target_model: Any = config.pop("target_model", None)
        self.params: dict[str, Any] = self.resolve(config)

        # Seed rule: a seed is accepted for any algorithm, but a
        # deterministic one has no draw, so it is stored as None; a stochastic
        # one requires a seed outright (only then does a draw exist).
        if self.stochastic and seed is None:
            raise ValueError(
                f"{self.algorithm}: a stochastic algorithm requires a seed; set "
                "'seed' in the transformation config (deterministic algorithms record None)"
            )
        self.seed: int | None = int(seed) if self.stochastic else None

        # Exactly the slot named by the role must be filled.
        needs = {"tool": self.tool_model, "target": self.target_model}
        for role, model in needs.items():
            if (model is not None) != (self.model_role == role):
                raise ValueError(
                    f"{self.algorithm}: model_role={self.model_role!r} but {role}_model="
                    f"{getattr(model, 'name', model)!r}; exactly the slot named by "
                    f"model_role must be filled"
                )
        if self.model_role is not None:
            check_model(self.algorithm, self.model_role, needs[self.model_role], self.model_methods)

    def _family(self) -> str:
        """The effective family: the declared one, or the role-derived default.

        Returns:
            ``self.family`` when the algorithm declares one, else
            ``FAMILY_BY_ROLE[model_role]``.
        """
        return self.family or FAMILY_BY_ROLE[self.model_role]

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
        """Complete the parameters, validating them through the registered ``Config``.

        The algorithm's registered pydantic schema (``self.Config``) is the
        only declaration of parameters and defaults: required parameters,
        ranges and unknown fields are refused here, defaults are filled by the
        schema, and the resolved parameters come out of ``model_dump`` — there
        is no second hand-maintained parameter table.

        Args:
            given: The raw parameters from the config.

        Returns:
            The parameters, complete, with every declared default filled in.

        Raises:
            ValueError: If an unknown parameter is given or a required one is
                missing (validated against ``self.Config``).
        """
        config_type = getattr(type(self), "Config", None)
        if config_type is None:
            return dict(given)
        # The config schema also carries the job-entry keys (name, type, seed,
        # storage, columns); only the algorithm's own parameters reach the row.
        job_keys = ("name", "type", "seed", "storage", "columns")
        validated = config_type.model_validate(given)
        return {k: v for k, v in validated.model_dump().items() if k not in job_keys}

    def rngs(self, keys: list[str]) -> list[np.random.Generator] | None:
        """Yield one generator per output, seeded by seed, params and parent id(s).

        Seeding from the output's parent ids makes a draw independent of both
        the batch and the order of the run. The resolved ``params`` are hashed
        in too, so the swept variants of one parent (one instance per value,
        same seed) each draw their own randomness.

        Args:
            keys: One key per output (the parent id, or the joined parent ids
                for n-ary), used to derive each generator's seed.

        Returns:
            One ``np.random.Generator`` per key, or ``None`` when the algorithm
            is deterministic.
        """
        if not self.stochastic:
            return None
        params_key = json.dumps(self.params, sort_keys=True, default=str)
        return [
            np.random.default_rng(
                [
                    self.seed,
                    int.from_bytes(hashlib.sha1(f"{k}::{params_key}".encode()).digest()[:8], "big"),
                ]
            )
            for k in keys
        ]

    def describe(self) -> dict[str, Any]:
        """Return the declaration-level fields every output row carries.

        Returns:
            A dictionary with ``algorithm``, ``family``, ``arity``,
            ``reversible``, ``params``, ``seed``, ``tool_model`` and
            ``target_model`` (models lowered to their ``name``).
        """
        return {
            "algorithm": self.algorithm,
            "family": self._family(),
            "arity": self.arity,
            "reversible": self.reversible,
            "params": dict(self.params),
            "seed": self.seed,
            "tool_model": getattr(self.tool_model, "name", None),
            "target_model": getattr(self.target_model, "name", None),
        }