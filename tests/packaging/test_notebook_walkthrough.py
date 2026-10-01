"""The walkthrough notebook, executed against an installed index.

The expensive end of the packaging suite: a fresh venv gets ``kcai-data-sampling`` with
its ``notebook`` extra from a real index, and the published walkthrough notebook then
runs in it -- procedural pipeline, LaMa inpainting behind the ``lama_inpaint`` entry
point, and FGSM against the user's own YOLO adapter.

Marked ``packaging_notebook`` and *not* ``packaging``, so the wheels-based
``nox -s test_packaging`` never collects it. Long by nature (a few hundred MB of wheels,
a 197 MB checkpoint, real CPU inference), so it is opt-in behind
``nox -s test_packaging_notebook`` and never runs in the default suite.

Two things are deliberate. The notebook is executed in the repository as-is, so it reads
``examples/config/*.yaml``, ``examples/adapters/yolo_target.py`` and the comma10k sample
-- every one of which lives under a git-ignored path, so the tree stays clean. And
nbclient is installed into the venv as test-only tooling rather than added to the
published ``notebook`` extra, which means this proves the extra is sufficient *except*
for an executor, a notebook author brings their own.
"""

from pathlib import Path
import subprocess

import pytest

from tests.fixtures.packaging_fixtures import (
    WORKSPACE_ROOT,
    create_venv,
    extra_requirements_for,
    install_from_index,
    install_specs,
    published_version,
)

pytestmark = pytest.mark.packaging_notebook

#: Where the walkthrough's inputs live; all git-ignored, all regenerated on demand.
SAMPLE_DIR = WORKSPACE_ROOT / "examples" / "data" / "comma10k_sample"
FETCH_SCRIPT = WORKSPACE_ROOT / "scripts" / "fetch_comma10k_sample.py"
EXECUTE_SCRIPT = Path(__file__).parent / "scripts" / "execute_walkthrough.py"

#: Frames the walkthrough covers; the fetch script pulls exactly these.
EXPECTED_FRAMES = 10


@pytest.fixture(scope="module")
def index_source(request: pytest.FixtureRequest) -> str:
    """Which index to install from, required rather than assumed.

    Args:
        request: The pytest request object.

    Returns:
        ``"testpypi"`` or ``"pypi"``.
    """
    source = request.config.getoption("--index-source")
    if source is None:
        pytest.skip("no --index-source given; use `nox -s test_packaging_notebook`")
    return str(source)


def _samples_present() -> bool:
    """Whether the comma10k sample is already on disk.

    Returns:
        True when the sidecar and every frame are present.
    """
    if not (SAMPLE_DIR / "samples.parquet").is_file():
        return False
    frames = list((SAMPLE_DIR / "imgs").glob("*.png")) if (SAMPLE_DIR / "imgs").is_dir() else []
    return len(frames) >= EXPECTED_FRAMES


def test_walkthrough_notebook_runs_from_an_installed_index(index_source: str, tmp_path: Path) -> None:
    """Install the notebook extra from an index, then run the walkthrough in it.

    Args:
        index_source: The index to install from.
        tmp_path: Per-test scratch, and the venv's home.
    """
    version = published_version()
    python = tmp_path / "venv" / "bin" / "python"
    venv = create_venv(tmp_path / "venv")
    install_from_index(
        venv,
        index_source,
        "all+notebook",
        version,
        extra_requirements=extra_requirements_for("all+notebook"),
    )
    # The executor is the runner's tool, not the extra's: see the module docstring.
    install_specs(venv, ["nbclient"])

    if not _samples_present():
        subprocess.run(
            [str(python), str(FETCH_SCRIPT)],
            cwd=WORKSPACE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    result = subprocess.run(
        [str(python), str(EXECUTE_SCRIPT), str(WORKSPACE_ROOT)],
        cwd=WORKSPACE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    print(result.stdout)
