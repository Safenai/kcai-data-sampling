"""What goes into a transformation run: one sample, and the data selection.

``DataSelection`` is the authority on what a *sample* is, in memory only. It is
not a batch: how samples are grouped for execution is chosen at run time and
changes nothing (batch invariance: outputs are independent of the batch
layout).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Sample:
    """One input sample.

    Attributes:
        id: Names the sample within its data selection.
        x: The sample itself, a numpy array of shape given by the selection's
            ``sample_axes`` (for images: ``(H, W, C)`` uint8).
        y: The annotation as the dataset gives it; the interface never reads
            it, and it passes through unchanged (label transport is postponed).
        source: The reader's provenance note (what it assembled the sample
            from), opaque to the interface, kept so that storage can write a
            selection that can be read again.
    """

    id: str
    x: np.ndarray
    y: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSelection:
    """What a run covers, and the authority on what a sample is.

    In memory only: ``dataset`` names the reader that assembled the samples.
    ``sample_axes`` and ``value_range`` are the space a sample lives in — its
    shape and its domain of values — which a unary transformation must
    preserve. A ``None`` value range leaves the range unchecked.

    Attributes:
        name: Unique name of this selection.
        dataset: Names the reader that assembled the samples.
        samples: The in-memory samples (a chunk of the source selection).
        sample_axes: Labels for the axis dimensions of ``x``.
        value_range: ``(low, high)`` domain of values, or ``None`` to skip
            range checks.
    """

    name: str
    dataset: str
    samples: list[Sample] = field(default_factory=list)
    sample_axes: tuple[str, ...] = ("channel", "height", "width")
    value_range: tuple[float, float] | None = (0.0, 1.0)

    def __len__(self) -> int:
        """Return the number of samples in the selection."""
        return len(self.samples)