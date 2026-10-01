"""Notebook-extra smoke: the umbrella's ``notebook`` extra, metadata-level only.

Runs inside the ``all+notebook`` venv, installed from an index as
``kcai-data-sampling[notebook]``: the extra's third-party dependencies are present,
the model-backed members it pulls in register their plugins in the *installed wheel
metadata*, and the umbrella still dispatches.

Deliberately cheap. No checkpoint download, no torch execution, and no real
dataset: frames are synthesized in memory and third-party modules are only probed
with ``find_spec``, so nothing heavy is actually imported -- notably ``ultralytics``,
whose ``cv2`` import needs a system OpenGL library this venv has no reason to carry.
The full walkthrough that *does* run real inference lives behind
``nox -s test_packaging_notebook``.
"""

import importlib.metadata
import importlib.util
import pathlib
import subprocess
import sys

#: Distributions the ``notebook`` extra must have installed, mapped to the top-level
#: module each one provides. Top-level names only: ``find_spec("a.b")`` imports its
#: parent package.
EXTRA_MODULES = {
    "pandas": "pandas",
    "python-pptx": "pptx",
    "ipykernel": "ipykernel",
    "ultralytics": "ultralytics",
}

#: Plugins the whole surface registers, resolved from installed metadata rather than
#: by importing the packages. ``inpaint``/``fgsm``/``lama_inpaint`` are the ones the
#: extra's two members contribute; the dataloader and writers come from ``-job``.
EXTRA_PLUGINS = {
    "kcai_data_sampling.models": ("lama_inpaint",),
    "kcai_data_sampling.transformations": ("horizontal_flip", "crop_resize", "inpaint", "fgsm"),
    "kcai_data_sampling.dataloaders": ("parquet",),
    "kcai_data_sampling.outputwriter": ("images", "parquet"),
}

#: The siblings re-run below, proving the extra install yields a working full surface.
SIBLING_SMOKES = ("smoke_core.py", "smoke_images.py", "smoke_job.py", "smoke_lama.py", "smoke_fgsm.py")


def _entry_points(group: str) -> dict[str, importlib.metadata.EntryPoint]:
    """The entry points a group resolves to, keyed by name.

    Args:
        group: The entry-point group name.

    Returns:
        A name-keyed mapping of every registered entry point.
    """
    return {entry.name: entry for entry in importlib.metadata.entry_points(group=group)}


def main() -> int:
    scratch = pathlib.Path(sys.argv[1])
    scratch.mkdir(parents=True, exist_ok=True)

    # The extra's own dependency metadata resolved to installed distributions.
    for distribution, module in EXTRA_MODULES.items():
        assert importlib.util.find_spec(module) is not None, f"[notebook] did not install {distribution}"

    # Every plugin is discoverable from the installed wheels -- this is what a config
    # like the walkthrough's `type: lama_inpaint` resolves through.
    for group, names in EXTRA_PLUGINS.items():
        found = _entry_points(group)
        missing = set(names) - set(found)
        assert not missing, f"[notebook] {group} is missing {sorted(missing)}"

    # The model plugin loads and exposes its declared surface (smoke_lama checks the
    # same on the class directly).
    tool_cls = _entry_points("kcai_data_sampling.models")["lama_inpaint"].load()
    assert tool_cls.__name__ == "LamaTool", tool_cls
    assert tool_cls.__module__.startswith("kcai_data_sampling_lama"), tool_cls.__module__
    assert tool_cls.channels == 3

    # The umbrella still dispatches under the extra install.
    from kcai_data_sampling.dependency import get_available_command

    commands = get_available_command()
    assert {"version", "list", "process"} <= set(commands), sorted(commands)

    for name in SIBLING_SMOKES:
        subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).parent / name), str(scratch)],
            check=True,
        )

    print("smoke_notebook ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
