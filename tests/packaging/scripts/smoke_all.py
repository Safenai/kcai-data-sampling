"""Umbrella smoke: core, images, job and the umbrella CLI installed together.

Runs inside the ``all`` scenario venv: the version/list/process dispatch is
present and working, and the three earlier smokes are re-run under this venv —
the full install recipe from the README works end to end. Note this scenario is
the wheel subset without ``-fgsm`` and ``-lama``, so ``version`` reports those
two as ``None``; see smoke_fgsm.py and smoke_lama.py for the installed case.
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
    # Every member is reported, installed or not. This scenario deliberately
    # omits -fgsm and -lama, so both read `None` here — that is the property
    # worth pinning. That they *resolve* under their own extras is covered by
    # smoke_fgsm.py and smoke_lama.py.
    for member in ("core", "job", "images", "fgsm", "lama"):
        assert f"{member}:" in text
    assert "fgsm: None" in text
    assert "lama: None" in text

    for name in ("smoke_core.py", "smoke_images.py", "smoke_job.py"):
        subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).parent / name), str(scratch)],
            check=True,
        )

    print("smoke_all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
