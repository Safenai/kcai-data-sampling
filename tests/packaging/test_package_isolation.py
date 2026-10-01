"""Packaging-isolation tests.

Each scenario installs only its wheel subset into a fresh venv and runs the
matching smoke script; every module root is then probed so each venv resolves
exactly its own subset — no sibling leaks in transitively, no missing pieces.
Deselected by default (``-m "not packaging"``); run with
``pytest -m packaging tests/packaging/``.
"""

from pathlib import Path

import pytest

from tests.fixtures.packaging_fixtures import (
    MODULES,
    SCENARIOS,
    SMOKE_SCRIPTS,
    create_venv,
    extra_requirements_for,
    install_wheels,
    probe_module,
    run_script,
)

pytestmark = pytest.mark.packaging


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_scenario_installs_independently_and_runs_its_smoke(
    scenario: str, built_wheels_dir: Path, tmp_path: Path
) -> None:
    """One fresh venv per scenario installs only that wheel subset.

    Installing only ``core`` (etc.) from built wheels — with every dependency
    resolved from the public index — proves each subset is independently
    installable, and the per-module probes prove the venv resolves exactly
    that subset: present packages import, absent siblings do not. The
    ``all+lama`` scenario additionally pulls the pinned CPU torch build from
    the PyTorch CPU index (the full opt-in recipe).
    """
    venv = create_venv(tmp_path / "venv")
    installed = install_wheels(venv, built_wheels_dir, scenario, extra_requirements=extra_requirements_for(scenario))
    assert len(installed) == len(SCENARIOS[scenario])

    for key, module in MODULES.items():
        assert probe_module(venv, module) == (key in SCENARIOS[scenario]), f"{scenario}: {module} isolation broken"

    script = Path(__file__).parent / "scripts" / SMOKE_SCRIPTS[scenario]
    result = run_script(venv, script, tmp_path / "scratch")
    assert result.returncode == 0, result.stderr
