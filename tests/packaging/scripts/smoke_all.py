"""Umbrella smoke: the full surface with all four packages installed.

Runs inside the ``all`` venv: the version/list/process dispatch is present and
working, and the three earlier smokes are re-run under this venv — the full
install recipe from the README works end to end.
"""

import contextlib
import io
import pathlib
import subprocess
import sys

from kcai_data_sampling.dependency import display_version, get_available_command


def main() -> int:
    scratch = pathlib.Path(sys.argv[1])
    commands = get_available_command()
    assert {"version", "list", "process"} <= set(commands)
    assert commands["process"] is not None

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        display_version()
    text = out.getvalue()
    assert "kcai-data-sampling:" in text
    assert "core:" in text
    assert "job:" in text
    assert "images:" in text

    for name in ("smoke_core.py", "smoke_images.py", "smoke_job.py"):
        subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).parent / name), str(scratch)],
            check=True,
        )

    print("smoke_all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
