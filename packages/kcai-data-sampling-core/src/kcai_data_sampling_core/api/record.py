"""What one output row carries."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Record:
    """One output row: what could not be recovered any other way.

    Retrievable things are referenced (`x′` by path, `x` and the annotation
    through the selection). Measurements (`δ`, PSNR…) are recomputed
    downstream: `params` and `seed` replay the run. What the algorithm
    declares about itself (`family`, `arity`, `reversible`) is frozen here,
    because a lookup from `algorithm` gives what the code says *now*.

    No judgement (the regime is a downstream verdict on the pair (output,
    task)), nothing about the annotation, and no id: an identity is the
    fields it would hash.
    """

    # lineage: the selection (a path) and the parent's id within it
    data_selection_path: str | None
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

    # where x′ landed, assigned by storage
    sample_path: str | None = None
