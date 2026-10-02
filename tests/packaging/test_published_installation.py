"""Published-install tests: one fresh venv per scenario, installed from an index.

Names the distributions instead of pointing at wheel files, so this exercises the
artifact actually uploaded: its declared dependencies, its extras metadata and its
entry points. Pick the source with ``--index-source`` — in practice through
``nox -s test_packaging_testpypi`` or ``nox -s test_packaging_pypi``, since both
also raise the per-test timeout that installing CPU torch outruns.

Marked ``packaging_index`` and deliberately *not* ``packaging``, so the wheels-based
``nox -s test_packaging`` never collects this module.
"""

from pathlib import Path

import pytest

from tests.fixtures.packaging_fixtures import (
    EXTRAS_SCENARIOS,
    MODULES,
    PUBLISHED_SCENARIOS,
    SMOKE_SCRIPTS,
    create_venv,
    extra_requirements_for,
    install_from_index,
    probe_module,
    published_version,
    run_script,
)

pytestmark = pytest.mark.packaging_index


@pytest.fixture(scope="module")
def index_source(request: pytest.FixtureRequest) -> str:
    """Which index to install from, required rather than assumed.

    Args:
        request: The pytest request object.

    Returns:
        ``"testpypi"`` or ``"pypi"``.

    Raises:
        pytest.skip.Exception: When the option is absent, naming the sessions to use.
    """
    source = request.config.getoption("--index-source")
    if source is None:
        pytest.skip("no --index-source given; use `nox -s test_packaging_testpypi` or `nox -s test_packaging_pypi`")
    return str(source)


@pytest.mark.parametrize("scenario", sorted(PUBLISHED_SCENARIOS))
def test_scenario_installs_from_index_and_runs_its_smoke(scenario: str, index_source: str, tmp_path: Path) -> None:
    """One fresh venv per scenario installs that subset from the published index.

    Pins every spec to one version, so the run tests the artifact that was uploaded
    rather than whatever else the index may serve later, and a re-run is
    reproducible. The per-module probes still hold: a venv must resolve exactly the
    scenario's own subset, so an absent sibling proves nothing leaks in transitively.
    For an extras scenario that is the whole assertion -- the install requests only
    ``kcai-data-sampling[<extra>]``, so each member that resolves got there through the
    extra's published dependency metadata.

    Args:
        scenario: The scenario name.
        index_source: The index to install from.
        tmp_path: Per-test scratch, and the venv's home.
    """
    version = published_version()
    venv = create_venv(tmp_path / "venv")
    install_from_index(venv, index_source, scenario, version, extra_requirements=extra_requirements_for(scenario))

    for key, module in MODULES.items():
        assert probe_module(venv, module) == (key in PUBLISHED_SCENARIOS[scenario]), f"{scenario}: {module} broken"

    script = Path(__file__).parent / "scripts" / SMOKE_SCRIPTS[scenario]
    # smoke_extras.py serves all six extras scenarios, so it takes the extra under test.
    extra_args = (scenario.removeprefix("umbrella+"),) if scenario in EXTRAS_SCENARIOS else ()
    result = run_script(venv, script, tmp_path / "scratch", *extra_args)
    assert result.returncode == 0, result.stderr
