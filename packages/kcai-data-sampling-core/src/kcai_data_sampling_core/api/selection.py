"""What goes in: one sample, and the data selection that says what a sample is."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Sample:
    """One input sample.

    `y` is the annotation as the dataset gives it; the interface never reads
    it. `id` names the sample within its data selection; `source` is the
    reader's provenance note (what it assembled the sample from), opaque to
    the interface and kept so that storage can write a selection that can
    be read again.
    """

    id: str
    x: np.ndarray
    y: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSelection:
    """What the campaign covers, and the authority on what a sample is.

    In memory only: ``dataset`` names the reader that assembled the samples.
    ``sample_axes`` and ``value_range`` are the space a sample lives in, its
    shape and its domain of values, which a unary transformation must
    preserve; ``None`` for a range leaves it unchecked. Not a batch: how
    samples are grouped for execution is chosen at run time and changes
    nothing.
    """

    name: str
    dataset: str
    samples: list[Sample] = field(default_factory=list)
    sample_axes: tuple[str, ...] = ("channel", "height", "width")
    value_range: tuple[float, float] | None = (0.0, 1.0)

    def __len__(self) -> int:
        return len(self.samples)
