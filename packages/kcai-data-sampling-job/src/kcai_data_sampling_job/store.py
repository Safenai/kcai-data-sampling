"""Where selections and outputs land, and how a row leads back.

The interface (``kcai-data-sampling-core``) works in memory and knows no
path. This package writes what it produced and adds the two columns only it
can fill: ``sample_path`` and ``data_selection_path``. A row on disk is an
``Output`` minus its bitmap plus those two.
"""

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_core.api.selection import DataSelection, Sample
from kcai_data_sampling_core.utils.images import load_image, quantize, save_image


class Store:
    """Two directories under one ``root``, which several campaigns can share:
    ``selections/`` and ``transformed_samples/`` (with ``rows.json``, a ledger
    of every output in it). Paths are kept as given."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.selections = self.root / "selections"
        self.outputs = self.root / "transformed_samples"

    # --- selections

    def selection_path(self, selection: DataSelection) -> Path:
        return self.selections / f"{selection.name}.json"

    def write_selection(self, selection: DataSelection) -> Path:
        """Written once. Same content again is idempotent; different content
        under the same name is refused, since rows produced against the first
        say what it was."""
        self.selections.mkdir(parents=True, exist_ok=True)
        path = self.selection_path(selection)
        content = json.dumps({
            "name": selection.name,
            "dataset": selection.dataset,
            "sample_axes": list(selection.sample_axes),
            "samples": [{"id": s.id, "source": s.source} for s in selection.samples],
        }, indent=2) + "\n"
        if path.exists() and path.read_text() != content:
            raise FileExistsError(
                f"{path} already holds a different selection. A selection is written once, "
                "rows that reference it say what it was, so give this one another name."
            )
        if not path.exists():
            path.write_text(content)
        return path

    # --- outputs

    @staticmethod
    def output_name(selection: DataSelection, output: Output) -> str:
        """``{selection}__{parent}__{algorithm}__{h6}__{c6}.png``: ``h6`` hashes
        ``params`` and ``seed``, ``c6`` the content as written. Distinct outputs
        never overwrite each other; two different outputs of one identity
        (non-deterministic kernels) land side by side; a bit-identical re-run
        lands on the same name."""
        h6 = hashlib.sha1(json.dumps({"params": output.params, "seed": output.seed}, sort_keys=True, default=str).encode())
        c6 = hashlib.sha1(np.ascontiguousarray(quantize(output.x)).tobytes())
        return f"{selection.name}__{output.parent_id}__{output.algorithm}__{h6.hexdigest()[:6]}__{c6.hexdigest()[:6]}.png"

    def write(self, selection: DataSelection, outputs: list[Output]) -> list[dict[str, Any]]:
        """Write the selection (once), each bitmap, and merge the rows into the
        ledger: a row replaces the one with the same path, any other is added.
        No annotation beside an output."""
        selection_path = self.write_selection(selection)
        self.outputs.mkdir(parents=True, exist_ok=True)

        rows = []
        for output in outputs:
            path = self.outputs / self.output_name(selection, output)
            save_image(path, output.x)
            rows.append({"data_selection_path": str(selection_path), **output.row(), "sample_path": str(path)})

        existing = self.rows()
        fresh = {r["sample_path"] for r in rows}
        merged = [r for r in existing if r["sample_path"] not in fresh] + rows
        (self.outputs / "rows.json").write_text(json.dumps(merged, indent=2, default=str))
        return rows

    def rows(self) -> list[dict[str, Any]]:
        """The ledger: every output in the directory, across runs and campaigns."""
        ledger = self.outputs / "rows.json"
        return json.loads(ledger.read_text()) if ledger.exists() else []

    # --- back from a row

    @staticmethod
    def read(row: dict[str, Any]) -> np.ndarray:
        """The bitmap of one row."""
        return load_image(row["sample_path"])

    @staticmethod
    def parent(row: dict[str, Any]) -> Sample:
        """The parent of one row: the selection by its path, the entry by id,
        the reader the selection names."""
        written = json.loads(Path(row["data_selection_path"]).read_text())
        entry = next(s for s in written["samples"] if s["id"] == row["parent_id"])
        reader = importlib.import_module(f"kcai_data_sampling_core.datasets.{written['dataset']}")
        return reader.load_sample(entry["id"], entry["source"])
