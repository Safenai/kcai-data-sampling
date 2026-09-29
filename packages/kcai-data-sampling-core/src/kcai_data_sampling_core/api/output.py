"""What one transformation output is, in memory.

The row carries declarations, what replays the run, and a derived ``id`` —
never a judgement (a regime is read afterwards, from the input/output pair),
never a measurement (delta is recomputed downstream), never a path
(storage/IO adds it), never an *arbitrary* key: the id is "the fields it
would hash", materialized for consumers and lineage.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


def output_identity(fields: dict[str, Any]) -> str:
    """Derive the deterministic id of one output from its row fields.

    Args:
        fields: The row fields (everything but the bitmap and the id itself).

    Returns:
        A 12-hex sha1 over the fields, stable across runs and environments
        (``sort_keys``, ``default=str``): the id is re-derivable by any
        consumer, and two rows with the same id share the same recipe.
    """
    identity = {k: v for k, v in fields.items() if k != "id"}
    return hashlib.sha1(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


@dataclass
class Output:
    """One output: the bitmap, and what could not be recovered any other way.

    Lineage is a reference, ``parent_id``, into the selection the run was
    given. Measurements (delta, PSNR, ...) are recomputed downstream: ``params``
    and ``seed`` replay the run. What the algorithm declares about itself
    (``family``, ``arity``, ``reversible``) is frozen here, because a lookup
    from ``algorithm`` gives what the code says *now*.

    Attributes:
        x: The generated sample itself, a numpy array.
        id: The derived identity: a 12-hex sha1 of this output's row fields.
        parent_id: Lineage, a reference into the run's input selection.
        algorithm: The transformation's identity string.
        family: ``"procedural"``, ``"generative"`` or ``"adversarial"``.
        arity: ``"unary"`` or ``"n-ary"``.
        reversible: Whether ``x`` is determined by ``x'``, as declared.
        params: The fully resolved algorithm parameters, as given in the config.
        seed: The random seed, ``None`` for deterministic algorithms.
        tool_model: Name of the tool model used, or ``None``.
        target_model: Name of the target model used, or ``None``.
    """

    x: np.ndarray
    parent_id: str

    # what the algorithm declares, frozen as it was when it ran
    algorithm: str
    family: str
    arity: str
    reversible: bool

    # what replays the run; every NULL is an assertion
    params: dict[str, Any]
    seed: int | None
    tool_model: str | None
    target_model: str | None

    @property
    def id(self) -> str:
        """The derived identity: a 12-hex sha1 of this output's row fields.

        Not a stored field: it is recomputed from ``row()``, so it cannot
        drift from the row it identifies.
        """
        return output_identity({k: v for k, v in vars(self).items() if k != "x"})

    def row(self) -> dict[str, Any]:
        """Return every field but the bitmap, plus the derived ``id``.

        Returns:
            A dictionary of the declarations and replay fields (everything
            except ``x``), with the derived ``id`` first and ``params`` kept
            as a ``dict`` (serialization to JSON is the writer's business).
        """
        fields = {k: v for k, v in vars(self).items() if k != "x"}
        return {**fields, "id": output_identity(fields)}
