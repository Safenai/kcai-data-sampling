"""Root pytest configuration and shared fixtures.

Session-scoped fixtures live here (``test_seed``) and the fixture helper
modules are re-exported so a test imports fixtures by name only (a single
import surface): ``tests.fixtures`` and ``tests.e2e.fixtures``.
"""

import pytest

from tests.e2e.fixtures.configs import (  # noqa: F401
    all_forms_config,
    standard_config,
    swept_fraction_config,
    write_yaml,
)
from tests.e2e.fixtures.paths import ledger_file, output_root, samples_dir  # noqa: F401
from tests.fixtures.data import (  # noqa: F401
    path_data,
    raw_bytes_data,
    synthetic_batch,
    synthetic_data_dir,
    synthetic_selection,
)
from tests.fixtures.packaging_fixtures import built_wheels_dir  # noqa: F401
from tests.fixtures.registries import cached_big_lama, registry_snapshot, stub_target, stub_tool  # noqa: F401
from tests.utils.seeds import get_test_seed


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the command-line options the packaging tests are configured with.

    Lives here rather than in ``tests/packaging/conftest.py`` because that file is
    only an *initial* conftest when pytest is invoked with ``tests/packaging`` as
    the exact argpath; this one is an ancestor of every argpath in use.
    """
    parser.addoption(
        "--index-source",
        choices=("testpypi", "pypi"),
        default=None,
        help=(
            "Which package index the published-install tests install from. "
            "Set it via 'nox -s test_packaging_testpypi' or "
            "'nox -s test_packaging_pypi' rather than by hand."
        ),
    )


@pytest.fixture(scope="session")
def test_seed() -> int:
    """Return the single seed that drives every fixture in the run.

    All randomized fixture builders (synthetic images, parquet sidecars,
    configs) are derived from this value, so a full ``pytest`` invocation is
    byte-reproducible for a given ``KCAI_TEST_SEED``.
    """
    return get_test_seed()
