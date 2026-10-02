"""Packaging-isolation fixtures: wheels, venv, install, run.

The smoke scripts install into **fresh** venvs from **built wheels** only —
each scenario proves its package subset installs independently on PyPI-indexed
deps (no private index, no sibling packages leaking in transitively). ``uv``
subprocesses always run with private-index environment variables stripped, so
the tests behave identically in the CI runner and this sandbox.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest

#: The workspace root — where the UV workspace, ``uv.toml`` and packages live.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

#: The six packages, workspace-name → distribution name.
PACKAGES = {
    "core": "kcai-data-sampling-core",
    "images": "kcai-data-sampling-images",
    "job": "kcai-data-sampling-job",
    "umbrella": "kcai-data-sampling",
    "lama": "kcai-data-sampling-lama",
    "fgsm": "kcai-data-sampling-fgsm",
}

#: The six packages, workspace-name → importable module root.
MODULES = {
    "core": "kcai_data_sampling_core",
    "images": "kcai_data_sampling_images",
    "job": "kcai_data_sampling_job",
    "umbrella": "kcai_data_sampling",
    "lama": "kcai_data_sampling_lama",
    "fgsm": "kcai_data_sampling_fgsm",
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
    # The adversarial recipe: every kcai package plus the numpy-only -fgsm
    # member. No extra index, no torch: the structural proof that the bare
    # five stay fgsm-free (``all`` never includes ``-fgsm``).
    "all+fgsm": ("core", "images", "job", "umbrella", "fgsm"),
}

#: The six extras the umbrella publishes, in ``pyproject.toml`` order.
EXTRAS = ("notebooks", "job", "images", "fgsm", "lama", "all")

#: Published-index scenarios that install **one** umbrella extra and nothing else
#: — the spec is only ``kcai-data-sampling[<extra>]``, never the siblings by name,
#: so the extra's own published dependency metadata is the only thing that can pull
#: them in. Each value is the module roots the install is expected to yield:
#: ``umbrella`` and ``core`` always (every member depends on core), plus the members
#: the extra names. Written out rather than derived from the extra's name, so the
#: expectation is an independent statement of what the metadata *should* say.
EXTRAS_SCENARIOS: dict[str, tuple[str, ...]] = {
    "umbrella+notebooks": ("core", "umbrella"),
    "umbrella+job": ("core", "job", "umbrella"),
    "umbrella+images": ("core", "images", "umbrella"),
    "umbrella+fgsm": ("core", "fgsm", "umbrella"),
    "umbrella+lama": ("core", "lama", "umbrella"),
    "umbrella+all": ("core", "fgsm", "images", "job", "lama", "umbrella"),
}

#: Scenario that installs the umbrella under ``all`` for the walkthrough: the notebook
#: exercises all three families, so the union extra is what it genuinely needs.
NOTEBOOK_SCENARIO = "umbrella+all"

SMOKE_SCRIPTS = {
    "core": "smoke_core.py",
    "core+images": "smoke_images.py",
    "core+job": "smoke_job.py",
    "all": "smoke_all.py",
    "all+lama": "smoke_lama.py",
    "all+fgsm": "smoke_fgsm.py",
    # Published-index runs only; unreachable from the wheels tests because those
    # iterate ``SCENARIOS``, which does not contain these keys.
    **dict.fromkeys(EXTRAS_SCENARIOS, "smoke_extras.py"),
}

#: The scenarios the published-index runs cover: the six wheel scenarios plus one per
#: extras scenario. Kept separate from ``SCENARIOS`` so ``nox -s test_packaging`` keeps
#: running exactly its own six.
PUBLISHED_SCENARIOS = {
    **SCENARIOS,
    **EXTRAS_SCENARIOS,
}

#: Extra names to request, keyed by scenario then package key. The extras scenarios
#: name *only* the umbrella spec, so nothing but the published extra metadata can pull
#: a member in.
PUBLISHED_EXTRAS: dict[str, dict[str, tuple[str, ...]]] = {
    **{f"umbrella+{extra}": {"umbrella": (extra,)} for extra in EXTRAS},
}

#: Scenarios whose resolution includes ``torch``, and so need the PyTorch CPU wheel
#: index rather than the multi-gigabyte CUDA build plain ``torch`` pulls on Linux.
#: ``umbrella+notebooks`` qualifies transitively: ``ultralytics`` depends on torch.
TORCH_SCENARIOS = frozenset({"all+lama", "umbrella+lama", "umbrella+all", "umbrella+notebooks"})

#: Environment variable that pins which published version the index tests install.
VERSION_ENV_VAR = "KCAI_PACKAGING_VERSION"

#: PEP 440 shape check for a normalized version: ``1.2.3``, ``0.1.0rc1``, ``1.0a2``.
_VERSION_PATTERN = re.compile(r"^\d+\.\d+(?:\.\d+)*(?:(?:a|b|rc|dev|post)\d+)*$")

#: The PyTorch CPU wheel index and the torch build the -lama recipe pins.
LAMA_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_PIN = "torch==2.14.0+cpu"

#: The published-package indexes the index-install scenarios install from.
PYPI_INDEX = "https://pypi.org/simple/"
TEST_PYPI_INDEX = "https://test.pypi.org/simple/"


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
    """Build every package wheel into a fresh out-dir (one-shot per session).

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
    """Session fixture: build every package wheel exactly once.

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


def extra_requirements_for(scenario: str) -> list[str] | None:
    """The extra ``uv pip install`` arguments a scenario needs beyond its wheels.

    Any scenario resolving ``-lama`` needs CPU torch from the PyTorch wheel index:
    plain ``torch`` on Linux would otherwise drag in the multi-gigabyte CUDA build.
    That covers ``all+lama``, ``umbrella+lama``, ``umbrella+all`` and
    ``umbrella+notebooks`` -- the last because ``ultralytics`` declares ``torch`` as
    its own dependency, so the notebooks extra pulls it transitively too.

    Args:
        scenario: One of ``SCENARIOS`` or ``PUBLISHED_SCENARIOS``.

    Returns:
        The extra arguments, or None when the scenario needs none.
    """
    if scenario in TORCH_SCENARIOS:
        return ["--index", LAMA_INDEX, TORCH_PIN]
    return None


def install_from_index(
    venv: Path,
    index_source: str,
    scenario: str,
    version: str,
    extra_requirements: list[str] | None = None,
) -> list[str]:
    """Install one scenario's packages from a published index, by distribution name.

    Names the distributions rather than pointing at wheel files, so this exercises the
    artifact actually uploaded: its declared dependencies, its extras metadata and its
    entry points.

    Args:
        venv: The fresh venv directory.
        index_source: ``"testpypi"`` or ``"pypi"``.
        scenario: One of ``PUBLISHED_SCENARIOS``.
        version: The pinned version, e.g. ``0.1.0rc1``.
        extra_requirements: Optional extra ``uv pip install`` arguments (e.g. the
            CPU torch index and pin from ``extra_requirements_for``).

    Returns:
        The distribution specs that were requested.

    Raises:
        ValueError: If ``index_source`` is not a known source.
        RuntimeError: If the install fails; carries uv's own diagnostics, which is the
            only useful part when a version was never uploaded or the index is
            unreachable.
    """
    if index_source == "testpypi":
        indexes = [TEST_PYPI_INDEX, PYPI_INDEX]
    elif index_source == "pypi":
        indexes = [PYPI_INDEX]
    else:
        raise ValueError(f"unknown index source {index_source!r}")

    # The specs to request, not the modules to expect: an extras scenario lists only
    # the umbrella in ``PUBLISHED_EXTRAS``, so the extra's own published metadata is
    # the only thing that can pull a member in. A wheel scenario has no extras, so its
    # whole subset is requested by name.
    extras = PUBLISHED_EXTRAS.get(scenario)
    keys = tuple(extras) if extras else PUBLISHED_SCENARIOS[scenario]
    specs = []
    for key in keys:
        names = extras.get(key) if extras else None
        spec = f"{PACKAGES[key]}=={version}"
        specs.append(f"{PACKAGES[key]}[{','.join(names)}]=={version}" if names else spec)

    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(venv / "bin" / "python"),
        # Without this, `[tool.uv.sources]` maps every kcai package to the local
        # workspace and would hijack a name-based install.
        "--no-config",
        "--prerelease",
        "allow",
        # uv's default `first-index` strategy refuses to fall through: seeing any
        # pydantic on Test PyPI (1.5a1) hides PyPI's 2.13 from the resolver. Both
        # indexes here are public and trusted and every kcai spec is pinned to an
        # exact version, so there is no dependency-confusion surface to protect.
        "--index-strategy",
        "unsafe-best-match",
    ]
    for index in indexes:
        command += ["--index", index]
    command += specs
    if extra_requirements:
        command += extra_requirements
    result = subprocess.run(
        command,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"uv could not install {scenario} from {index_source} at version {version}. "
            f"Command: {' '.join(command)}\n{result.stderr.strip()}"
        )
    return specs


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


def install_specs(venv: Path, specs: list[str], extra_requirements: list[str] | None = None) -> None:
    """Install named distributions into a venv, resolved from public PyPI.

    For test-only tooling that is not part of any kcai extra -- the notebook
    executor, for instance, which the harness needs but the ``all`` extra
    deliberately does not ship.

    Args:
        venv: The venv to install into.
        specs: Distribution specs, e.g. ``["nbclient"]``.
        extra_requirements: Optional extra ``uv pip install`` arguments.
    """
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(venv / "bin" / "python"),
        "--no-config",
        "--prerelease",
        "allow",
        "--index",
        PYPI_INDEX,
        *specs,
    ]
    if extra_requirements:
        command += extra_requirements
    subprocess.run(
        command,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=True,
    )


def run_script(venv: Path, script: Path, scratch: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a smoke script inside the venv with a scratch directory.

    Args:
        venv: The venv whose python runs the script.
        script: The smoke script path.
        scratch: Writable scratch directory handed to the script (``argv[1]``)
            for any files it needs to produce; never the source tree.
        *args: Anything after the scratch directory (``argv[2:]``) — ``smoke_extras.py``
            takes the extra name under test there, since one script serves all six
            extras scenarios.

    Returns:
        The completed subprocess (checked, output captured).
    """
    env = _subprocess_env()
    return subprocess.run(
        [str(venv / "bin" / "python"), str(script), str(scratch), *args],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _latest_version_tag() -> str | None:
    """The newest numeric-looking git tag, or None when there is none.

    Returns:
        The tag as written (``0.1.0-rc1``), or None.
    """
    result = subprocess.run(
        ["git", "describe", "--tags", "--match", "[0-9]*", "--abbrev=0"],
        cwd=WORKSPACE_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def published_version() -> str:
    """The published version under test, pinned explicitly and never guessed.

    Resolution order: ``KCAI_PACKAGING_VERSION``, else the newest numeric git tag.
    The tag is normalized with the same rule the publish workflows use
    (``sed 's/-//g'``), so ``0.1.0-rc1`` becomes ``0.1.0rc1`` — exactly the wheel
    name on the index.

    Returns:
        The pinned version, e.g. ``0.1.0rc1``.

    Raises:
        RuntimeError: If no version can be determined at all.
        ValueError: If the resolved value is not a plausible version.
    """
    raw = os.environ.get(VERSION_ENV_VAR) or _latest_version_tag()
    if not raw:
        raise RuntimeError(
            f"Cannot tell which published version to install: {VERSION_ENV_VAR} is unset and "
            "`git describe --tags --match '[0-9]*'` found no tag. Set the variable explicitly, "
            "for example KCAI_PACKAGING_VERSION=0.1.0rc1."
        )
    version = raw.strip().removeprefix("v").replace("-", "")
    if not _VERSION_PATTERN.match(version):
        raise ValueError(f"{VERSION_ENV_VAR}={raw!r} does not normalize to a valid version ({version!r})")
    return version


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
