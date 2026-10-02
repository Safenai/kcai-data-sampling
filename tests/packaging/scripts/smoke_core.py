"""Core-only smoke: a generic Batch and an Output round-trip, no siblings.

Runs inside a ``core``-only venv: imports the generic API and the runner,
hand-builds a generic batch and a local unary map (plain pydantic Config,
``extra="forbid"`` — the same surface attributes tests use) and asserts the
transform/ledger surface keeps working on the wheel-installed package.
"""

import sys

from kcai_data_sampling_core.api.selection import Batch
from kcai_data_sampling_core.api.unary import UnaryTransformation
from kcai_data_sampling_core.utils.runner import TransformationRunner
import numpy as np
from pydantic import BaseModel, ConfigDict


class AddOneConfig(BaseModel):
    """Local parameter schema with no parameters; nothing else is accepted."""

    model_config = ConfigDict(extra="forbid")


class AddOne(UnaryTransformation):
    """A core-usable unary map: ``x -> x + 1`` on a float generic batch."""

    algorithm = "add_one"
    family = "procedural"
    reversible = True
    Config = AddOneConfig

    def apply(self, xs, rngs=None):
        return xs + 1.0


def main() -> int:
    batch = Batch(
        name="smoke",
        dataset="smoke",
        ids=["a", "b", "c"],
        columns=None,
        data=np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2),
    )
    outputs = TransformationRunner(batch).run(AddOne({}))
    assert len(outputs) == 3
    assert [o.parent_id for o in outputs] == ["a", "b", "c"]
    assert outputs[0].x.max() == 4.0
    assert [o.algorithm for o in outputs] == ["add_one"] * 3
    assert outputs[0].seed is None
    print("smoke_core ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
