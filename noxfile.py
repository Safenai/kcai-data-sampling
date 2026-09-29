"""Nox sessions for building, testing, linting, and documenting the kcai data-sampling packages.

Mirrors ``dqm-ml-workspace/noxfile.py``: sessions are declared with ``nox_uv``
and carry ``uv_groups``, so each session's environment is created by ``uv``
from the matching ``[dependency-groups]`` in the root ``pyproject.toml``.
"""

from nox import Session, options, param, parametrize
from nox_uv import session

options.error_on_external_run = True
options.default_venv_backend = "uv"
options.sessions = ["lint", "spell", "test", "type_check"]


@session(
    uv_groups=["test"],
)
def test(s: Session) -> None:
    """Run the neutral test suite: unit, e2e and CLI tests, packaging / fgsm / lama excised.

    The default env stays torch-free; the opt-in surfaces are exercised by the
    dedicated sessions and measured by ``test_coverage``.
    """
    s.env["KCAI_TEST_SEED"] = "42"
    s.run(
        "pytest",
        "tests/unit",
        "tests/e2e",
        "tests/cli",
        "-m",
        "not packaging and not fgsm and not lama",
        *s.posargs,
    )


@session(
    uv_groups=["test", "test-fgsm", "test-lama"],
)
def test_coverage(s: Session) -> None:
    """Run every test (unit, e2e, CLI, opt-in fgsm/lama surfaces) with full coverage.

    The env includes the -fgsm and -lama members (CPU torch + ``big-lama.pt``),
    so every package in the workspace is measured here.
    """
    s.env["KCAI_TEST_SEED"] = "42"
    s.run(
        "pytest",
        "--cov=packages/kcai-data-sampling-core/src",
        "--cov=packages/kcai-data-sampling-images/src",
        "--cov=packages/kcai-data-sampling-images-lama/src",
        "--cov=packages/kcai-data-sampling-fgsm/src",
        "--cov=packages/kcai-data-sampling-job/src",
        "--cov=packages/kcai-data-sampling/src",
        "--cov-report=html:docs/reports/htmlcov",
        "--cov-report=json:docs/reports/coverage.json",
        "--cov-report=term",
        "--cov-fail-under=90",
        "--html=docs/reports/pytest/pytest_report.html",
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
    """Run only the opt-in (torch-gated) lama tests in the -lama env."""
    s.env["KCAI_TEST_SEED"] = "42"
    s.run(
        "pytest",
        "tests/unit",
        "tests/e2e",
        "tests/cli",
        "-m",
        "lama",
        *s.posargs,
    )


@session(
    uv_groups=["test-fgsm"],
)
def test_fgsm(s: Session) -> None:
    """Run only the opt-in (importorskip-gated) fgsm tests in the -fgsm env."""
    s.env["KCAI_TEST_SEED"] = "42"
    s.run(
        "pytest",
        "tests/unit",
        "tests/e2e",
        "tests/cli",
        "-m",
        "fgsm",
        *s.posargs,
    )


@session(
    python=["3.11", "3.12", "3.13", "3.14"],
    uv_groups=["test"],
)
def compatibility(s: Session) -> None:
    """Run unit and CLI tests across Python 3.11-3.14 to verify cross-version compatibility."""
    s.env["KCAI_TEST_SEED"] = "42"
    s.run(
        "pytest",
        "tests/unit",
        "tests/cli",
        "-m",
        "not packaging",
        *s.posargs,
    )


# For some sessions, set venv_backend="none" to simply execute scripts within the existing outer
# uv-generated virtual environment, rather than have nox create a new one for each session. This
# makes commonly repeated sessions execute faster.


@session(venv_backend="none")
@parametrize(
    "command",
    [
        param(
            [
                "ruff",
                "check",
                "packages",
                "tests",
                "--select",
                "I",
                # Also remove unused imports.
                "--select",
                "F401",
                "--extend-fixable",
                "F401",
                "--fix",
            ],
            id="sort_imports",
        ),
        param(
            [
                "ruff",
                "format",
                "packages",
                "tests",
            ],
            id="format",
        ),
    ],
)
def fmt(s: Session, command: list[str]) -> None:
    """Auto-format code: (1) sort imports (I, F401), (2) format via ruff. Runs in the outer venv for speed."""
    s.run(*command)


@session(uv_groups=["lint"])
def lint(s: Session) -> None:
    """Check code quality — ruff lint + format --check on packages and tests (read-only, no fixes)."""
    s.run("ruff", "check", "packages")
    s.run("ruff", "format", "--check", "packages")
    s.run("ruff", "check", "tests")
    s.run("ruff", "format", "--check", "tests")


@session(uv_groups=["type_check"])
def type_check(s: Session) -> None:
    """Run mypy static type checking on all 6 sub-packages."""
    s.run("mypy", "packages/kcai-data-sampling-job")
    s.run("mypy", "packages/kcai-data-sampling-core")
    s.run("mypy", "packages/kcai-data-sampling-images")
    s.run("mypy", "packages/kcai-data-sampling-images-lama")
    s.run("mypy", "packages/kcai-data-sampling-fgsm")
    s.run("mypy", "packages/kcai-data-sampling")


# Environment variable needed for mkdocstrings-python to locate source files.
doc_env = {"PYTHONPATH": "packages"}


@session(
    python=["3.12"],
    uv_groups=["docs"],
)
def docs_serve(s: Session) -> None:
    """Serve mkdocs documentation locally for development preview."""
    s.run(
        "mkdocs",
        "serve",
        *s.posargs,
        env=doc_env,
    )


@session(
    python=["3.12"],
    uv_groups=["docs"],
)
def docs_github_pages(s: Session) -> None:
    """Deploy mkdocs site to GitHub Pages via gh-deploy."""
    s.run("mkdocs", "gh-deploy", "--force", env=doc_env)


# Install only main dependencies for the license report.
@session(uv_groups=["licenses"], uv_no_install_project=True)
def licenses(s: Session) -> None:
    """Generate a third-party license report via pip-licenses. Installs only main dependencies."""
    s.run("pip-licenses", *s.posargs)


@session(uv_groups=["complexity"])
def complexity(s: Session) -> None:
    """Check cyclomatic complexity of source code (packages/). Fails if any function exceeds 15."""
    s.run("complexipy", "packages", "--max-complexity-allowed", "15")


@session(uv_groups=["complexity"])
def test_complexity(s: Session) -> None:
    """Check cyclomatic complexity of test code (tests/). Fails if any function exceeds 15."""
    s.run("complexipy", "tests", "--max-complexity-allowed", "15")


@session(uv_groups=["spell"])
def spell(s: Session) -> None:
    """Spell-check the entire repository using cspell."""
    s.run("cspell", "lint", ".", *s.posargs)
