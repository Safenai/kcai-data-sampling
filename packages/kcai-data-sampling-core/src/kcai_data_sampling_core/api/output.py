"""What one transformation output is, in memory."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Output:
    """One output: the bitmap, and what could not be recovered any other way.

    Lineage is a reference, ``parent_id``, into the selection the run was
    given. Measurements (`δ`, PSNR…) are recomputed downstream: `params` and
    `seed` replay the run. What the algorithm declares about itself
    (`family`, `arity`, `reversible`) is frozen here, because a lookup from
    `algorithm` gives what the code says *now*.

    No path of any kind: where this lands is the storage module's business,
    and it adds its own columns when it writes. No judgement (the regime is
    a downstream verdict), nothing about the annotation, and no id: an
    identity is the fields it would hash.
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
        """Every field but the bitmap."""
        return {k: v for k, v in vars(self).items() if k != "x"}
