"""What one transformation output is, in memory.

The row carries declarations and what replays the run — never a judgement
(a regime is read afterwards, from the input/output pair), never a measurement
(delta is recomputed downstream), never a path (storage/IO adds it), and no id
(an identity is the fields it would hash).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


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
        parent_id: Lineage, a reference into the run's input selection.
        algorithm: The transformation's identity string.
        family: ``"procedural"``, ``"generative"`` or ``"adversarial"``.
        arity: ``"unary"`` or ``"n-ary"``.
        reversible: Whether ``x`` is determined by ``x′``, as declared.
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

    def row(self) -> dict[str, Any]:
        """Return every field but the bitmap, as a row-ready dict.

        Returns:
            A dictionary of the declarations and replay fields (everything
            except ``x``), with ``params`` kept as a ``dict`` (serialization
            to JSON is the writer's business).
        """
        return {k: v for k, v in vars(self).items() if k != "x"}