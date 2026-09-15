"""What goes in: one sample, and the data selection that says what a sample is."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Sample:
    """One input sample.

    `y` is the annotation as the dataset gives it; the interface never reads
    it. `id` names the sample within its data selection, `source` says what
    it was assembled from, one image file, or four frames on a time axis.
    """

    id: str
    x: np.ndarray
    y: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSelection:
    """What the campaign covers, and the authority on what a sample is.

    Each sample has an ``id`` and a ``source`` (what it was assembled from);
    ``dataset`` names the reader that assembles it; ``sample_axes`` is the
    shape of a sample. Written to disk before anything runs, so that a row
    can reference it by path. Not a batch: how samples are grouped for
    execution is chosen at run time and changes nothing about the rows.
    """

    name: str
    dataset: str
    samples: list[Sample] = field(default_factory=list)
    sample_axes: tuple[str, ...] = ("channel", "height", "width")
    path: str | None = None  # set by `save`

    def __len__(self) -> int:
        return len(self.samples)

    def to_dict(self) -> dict[str, Any]:
        """Everything but the pixels, enough to assemble every sample again."""
        return {
            "name": self.name,
            "dataset": self.dataset,
            "sample_axes": list(self.sample_axes),
            "samples": [{"id": s.id, "source": s.source} for s in self.samples],
        }

    def save(self, directory: Path | str) -> Path:
        """Write ``<directory>/<name>.json``, once.

        Paths are kept as given. Same content again is idempotent; different
        content under the same name is refused, since rows produced against
        the first say what it was.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        content = json.dumps(self.to_dict(), indent=2) + "\n"

        if path.exists() and path.read_text() != content:
            raise FileExistsError(
                f"{path} already holds a different selection. A selection is written once, "
                "rows that reference it say what it was, so give this one another name."
            )
        if not path.exists():
            path.write_text(content)
        self.path = str(path)
        return path
