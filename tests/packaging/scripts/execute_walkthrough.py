"""Execute the real walkthrough notebook against an installed index, then verify it.

Runs inside the scenario venv, because nbclient, pandas, pyarrow and Pillow are all
things only that venv has. The notebook itself lives at
``examples/notebooks/walkthrough.ipynb`` and is a *generated* docs artifact, so this
asserts on what it writes to disk rather than on the prose it prints.

Three families are exercised end to end: the procedural pipeline through the job CLI,
LaMa inpainting behind the ``lama_inpaint`` entry point, and FGSM against the user's own
YOLO adapter loaded through the ``python`` model route. Any failure surfaces as a
CellExecutionError, which is the strongest signal here -- the notebook's own cells
already index into the ledgers, so a missing campaign row raises inside the kernel.

Kernelspec isolation: ``JUPYTER_DATA_DIR`` is pinned to the venv so no user- or
system-wide ``python3`` kernelspec can start a different interpreter than the one whose
wheels were just installed.
"""

import os
import pathlib
import sys

#: The loader/selection name every walkthrough config uses, hence the campaign
#: directory they all write into.
CAMPAIGN = "comma10k"

#: How many rows to compare against their source frames. Enough to prove the attack
#: moved pixels without decoding thirty 1208x1928 PNGs.
CHECKED_ROWS = 3


def _ledger_rows(outputs: pathlib.Path) -> list[dict]:
    """The campaign ledger as plain dicts.

    Each config re-uses the ``comma10k`` selection, so the ledger holds the last
    campaign that ran -- the adversarial one. Its presence is what this checks.

    Args:
        outputs: The ``examples/outputs`` directory.

    Returns:
        One dict per ledger row.
    """
    import pandas as pd

    ledger = pd.read_parquet(outputs / CAMPAIGN / "ledger.parquet")
    assert not ledger.empty, "the notebook wrote an empty ledger"
    return ledger.to_dict("records")


def _max_delta(row: dict, sample_dir: pathlib.Path, payloads: pathlib.Path) -> float:
    """The largest per-pixel difference between an output payload and its source frame.

    Args:
        row: A ledger row.
        sample_dir: The comma10k sample directory holding the originals.
        payloads: The campaign's payload directory.

    Returns:
        The maximum absolute difference over the frame.
    """
    import numpy as np
    from PIL import Image

    source = np.asarray(Image.open(sample_dir / str(row["source_path"])).convert("RGBA"), dtype=np.int16)
    output = np.asarray(Image.open(payloads / str(row["artifact"])).convert("RGBA"), dtype=np.int16)
    assert output.shape == source.shape, f"{row['artifact']}: {output.shape} vs source {source.shape}"
    return float(np.abs(output - source).max())


def main() -> int:
    repo = pathlib.Path(sys.argv[1]).resolve()
    notebook_path = repo / "examples" / "notebooks" / "walkthrough.ipynb"
    assert notebook_path.is_file(), notebook_path

    sample_dir = repo / "examples" / "data" / "comma10k_sample"
    assert (sample_dir / "samples.parquet").is_file(), f"missing {sample_dir}; run scripts/fetch_comma10k_sample.py"

    # Isolate the kernelspec search to this venv before nbclient starts a kernel: the
    # kernel inherits this environment, so a stray user-level `python3` kernelspec
    # cannot hijack the interpreter under test.
    os.environ["JUPYTER_DATA_DIR"] = str(pathlib.Path(sys.prefix) / "share" / "jupyter")
    os.environ.pop("JUPYTER_PATH", None)

    from nbclient import NotebookClient
    import nbformat

    notebook = nbformat.read(notebook_path, as_version=4)
    # The kernel's cwd must be the notebook's own directory: the setup cell imports a
    # sibling `display` module *before* it walks up to the repo root.
    client = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=None,
        allow_errors=False,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
    print(f"executed {notebook_path.name}: {len(notebook.cells)} cells, no cell errors")

    outputs = repo / "examples" / "outputs"
    rows = _ledger_rows(outputs)
    attacks = [row for row in rows if row["algorithm"] == "fgsm"]
    seen = sorted({str(row["algorithm"]) for row in rows})
    assert attacks, f"no fgsm rows in the ledger ({len(rows)} rows, algorithms={seen})"
    assert all(row["family"] == "adversarial" for row in attacks)
    assert all(row["target_model"] for row in attacks), "fgsm rows must name the target model they attacked"

    # Join the outputs back to their source frames through the sample sidecar: the
    # loader records `parent_id` as the sample `id`, and `path` is relative to the
    # sample directory.
    import pandas as pd

    samples = pd.read_parquet(sample_dir / "samples.parquet").set_index("id")
    payloads = outputs / CAMPAIGN / "payloads"
    deltas = []
    for row in attacks[:CHECKED_ROWS]:
        row["source_path"] = str(samples.loc[row["parent_id"], "path"])
        deltas.append(_max_delta(row, sample_dir, payloads))
    assert max(deltas) > 0, f"fgsm left every checked frame untouched (deltas={deltas})"
    print(f"fgsm moved pixels on {len(deltas)} frames: max |delta| up to {max(deltas):g}")

    print("execute_walkthrough ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
