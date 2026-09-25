"""Packaging-isolation fixtures: wheels, venv, install, run.

The smoke scripts install into **fresh** venvs from **built wheels** only —
each scenario proves its package subset installs independently on PyPI-indexed
deps (no private index, no sibling packages leaking in transitively). ``uv``
subprocesses always run with private-index environment variables stripped, so
the tests behave identically in the CI runner and this sandbox.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

#: The workspace root — where the UV workspace, ``uv.toml`` and packages live.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

#: The five packages, workspace-name → distribution name.
PACKAGES = {
    "core": "kcai-data-sampling-core",
    "images": "kcai-data-sampling-images",
    "job": "kcai-data-sampling-job",
    "umbrella": "kcai-data-sampling",
    "lama": "kcai-data-sampling-images-lama",
}

#: The five packages, workspace-name → importable module root.
MODULES = {
    "core": "kcai_data_sampling_core",
    "images": "kcai_data_sampling_images",
    "job": "kcai_data_sampling_job",
    "umbrella": "kcai_data_sampling",
    "lama": "kcai_data_sampling_images_lama",
}

SCENARIOS = {
    "core": ("core",),
    "core+images": ("core", "images"),
    "core+job": ("core", "job"),
    "all": ("core", "images", "job", "umbrella"),
    # The full user recipe: every kcai package plus the opt-in -lama member and
    # CPU torch. torch is pulled from the PyTorch CPU wheel index, pinned
    # to the same `+cpu` build the workspace's test-lama env resolves.
    "all+lama": ("core", "images", "job", "umbrella", "lama"),
}

SMOKE_SCRIPTS = {
    "core": "smoke_core.py",
    "core+images": "smoke_images.py",
    "core+job": "smoke_job.py",
    "all": "smoke_all.py",
    "all+lama": "smoke_lama.py",
}

#: The PyTorch CPU wheel index and the torch build the -lama recipe pins.
LAMA_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_PIN = "torch==2.14.0+cpu"


def _subprocess_env() -> dict[str, str]:
    """The environment for every ``uv``/script subprocess.

    Private index / token env vars are stripped so resolution uses only the
    registered public index (``uv.toml``), never a GitLab-hosted one.

    Returns:
        A copy of ``os.environ`` without private index variables.
    """
    env = os.environ.copy()
    for key in ("UV_INDEX", "UV_INDEX_URL", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"):
        env.pop(key, None)
    return env


def build_wheels(out: Path) -> Path:
    """Build the four wheels into a fresh out-dir (one-shot per session).

    Args:
        out: The (empty or absent) output directory for the wheels.

    Returns:
        The directory holding ``<dist>-<version>.whl`` for every package.
    """
    if out.exists():
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True)
    env = _subprocess_env()
    for distribution in PACKAGES.values():
        subprocess.run(
            ["uv", "build", "--package", distribution, "--wheel", "--out-dir", str(out)],
            cwd=WORKSPACE_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    return out


@pytest.fixture(scope="session")
def built_wheels_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session fixture: build all four wheels exactly once.

    Args:
        tmp_path_factory: Pytest session temp factory.

    Returns:
        The wheels out-dir.
    """
    return build_wheels(tmp_path_factory.mktemp("wheels"))


def create_venv(venv_dir: Path) -> Path:
    """Create a fresh venv with the workspace's supported Python.

    Args:
        venv_dir: Where the venv goes.

    Returns:
        The venv directory.
    """
    env = _subprocess_env()
    subprocess.run(
        ["uv", "venv", "--python", "3.12", str(venv_dir)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return venv_dir


def install_wheels(
    venv: Path,
    wheels_dir: Path,
    scenario: str,
    extra_requirements: list[str] | None = None,
) -> list[Path]:
    """Install one scenario's wheel subset into a venv, resolving deps publicly.

    Args:
        venv: The fresh venv directory.
        wheels_dir: Directory with the built wheels.
        scenario: One of ``SCENARIOS``.
        extra_requirements: Optional extra ``uv pip install`` arguments (e.g.
            ``["--extra-index-url", LAMA_INDEX, TORCH_PIN]`` for the ``all+lama``
            scenario's CPU torch).

    Returns:
        The installed wheel files.
    """
    wheels = []
    for key in SCENARIOS[scenario]:
        distribution = PACKAGES[key].replace("-", "_")
        wheels.append(next(wheels_dir.glob(f"{distribution}-*.whl")))
    env = _subprocess_env()
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(venv / "bin" / "python"),
        *map(str, wheels),
    ]
    if extra_requirements:
        command += extra_requirements
    subprocess.run(
        command,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return wheels


def run_script(venv: Path, script: Path, scratch: Path) -> subprocess.CompletedProcess[str]:
    """Run a smoke script inside the venv with a scratch directory.

    Args:
        venv: The venv whose python runs the script.
        script: The smoke script path.
        scratch: Writable scratch directory handed to the script (``argv[1]``)
            for any files it needs to produce; never the source tree.

    Returns:
        The completed subprocess (checked, output captured).
    """
    env = _subprocess_env()
    return subprocess.run(
        [str(venv / "bin" / "python"), str(script), str(scratch)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def probe_module(venv: Path, module: str) -> bool:
    """Whether ``module`` resolves (``find_spec``) inside the venv.

    The isolation half of the scenario: a venv must resolve exactly its own
    package subset — absent siblings prove no wheel leaks them transitively.

    Args:
        venv: The venv to probe.
        module: The module root, e.g. ``kcai_data_sampling_images``.

    Returns:
        True when the module is importable in that venv.
    """
    env = _subprocess_env()
    result = subprocess.run(
        [
            str(venv / "bin" / "python"),
            "-c",
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
            module,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0