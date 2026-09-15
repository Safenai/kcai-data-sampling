"""Images in and out. Memory is CHW float32 in [0, 1]; disk is PNG."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from kcai_data_sampling_core.api.record import Record


def load_image(path: Path | str) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("RGB"), dtype="float32") / 255.0
    return array.transpose(2, 0, 1)


def quantize(x: np.ndarray) -> np.ndarray:
    """The HWC uint8 array a PNG will hold."""
    return (np.clip(x, 0, 1).transpose(1, 2, 0) * 255).round().astype("uint8")


def save_image(path: Path | str, x: np.ndarray) -> None:
    Image.fromarray(quantize(x)).save(path)


def output_name(record: Record, x_prime: np.ndarray) -> str:
    """``{selection}__{parent}__{algorithm}__{h6}__{c6}.png``.

    ``h6`` hashes ``params`` and ``seed``, ``c6`` the content as written. The
    first four parts keep distinct outputs from overwriting each other; the
    content digest keeps two different outputs of one identity (non-
    deterministic kernels) side by side, while a bit-identical re-run lands
    on the same name. Grouping keys for storage only, the row carries no id.
    """
    selection = Path(record.data_selection_path).stem if record.data_selection_path else "unsaved"
    h6 = hashlib.sha1(json.dumps({"params": record.params, "seed": record.seed}, sort_keys=True, default=str).encode())
    c6 = hashlib.sha1(np.ascontiguousarray(quantize(x_prime)).tobytes())
    return f"{selection}__{record.parent_id}__{record.algorithm}__{h6.hexdigest()[:6]}__{c6.hexdigest()[:6]}.png"


def save_outputs(directory: Path | str, results: list[tuple[np.ndarray, Record]]) -> list[dict]:
    """Write each x′ and merge the rows into ``rows.json``, a ledger of every
    output in the directory: a row replaces the one with the same path, any
    other is added. Nothing else is written, no annotation beside an output."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for x_prime, record in results:
        path = directory / output_name(record, x_prime)
        save_image(path, x_prime)
        record.sample_path = str(path)
        rows.append(asdict(record))

    ledger = directory / "rows.json"
    existing = json.loads(ledger.read_text()) if ledger.exists() else []
    fresh = {r["sample_path"] for r in rows}
    merged = [r for r in existing if r["sample_path"] not in fresh] + rows
    ledger.write_text(json.dumps(merged, indent=2, default=str))
    return rows
