"""Nox sessions for building, testing, linting, and documenting the kcai data-sampling packages.

Mirrors ``dqm-ml-workspace/noxfile.py``: sessions are declared with ``nox_uv``
and carry ``uv_groups``, so each session's environment is created by ``uv``
from the matching ``[dependency-groups]`` in the root ``pyproject.toml``.
"""

from nox import Session, options
from nox_uv import session

options.error_on_external_run = True
options.default_venv_backend = "uv"
options.sessions = ["test"]


@session(
    uv_groups=["test"],
)
def test(s: Session) -> None:
    """Run the test suite: unit, e2e and CLI tests (packaging excluded by default)."""
    s.env["KCAI_TEST_SEED"] = "42"
    s.run(
        "pytest",
        "tests/unit",
        "tests/e2e",
        "tests/cli",
        "-m",
        "not packaging",
        *s.posargs,
    )


@session(
    uv_groups=["test"],
)
def test_packaging(s: Session) -> None:
    """Run the package-isolation tests: fresh venvs, four wheels, smoke scripts."""
    s.env["KCAI_TEST_SEED"] = "42"
    s.run(
        "pytest",
        "tests/packaging",
        "-m",
        "packaging",
        *s.posargs,
    )


@session(
    uv_groups=["test-lama"],
)
def test_lama(s: Session) -> None:
    """Run the same default suite in the -lama + torch env (opt-in surface active)."""
    s.env["KCAI_TEST_SEED"] = "42"
    s.run(
        "pytest",
        "tests/unit",
        "tests/e2e",
        "tests/cli",
        "-m",
        "not packaging",
        *s.posargs,
    )