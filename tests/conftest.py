"""Root pytest configuration and shared fixtures.

Session-scoped fixtures live here (``test_seed``) and the fixture helper
modules are re-exported so a test imports fixtures by name only (a single
import surface): ``tests.fixtures`` and ``tests.e2e.fixtures``.
"""

import pytest

from tests.fixtures.data import (  # noqa: F401
    path_data,
    raw_bytes_data,
    synthetic_batch,
    synthetic_data_dir,
    synthetic_selection,
)
from tests.fixtures.registries import (  # noqa: F401
    cached_big_lama,
    registry_snapshot,
    stub_target,
    stub_tool,
)
from tests.e2e.fixtures.configs import (  # noqa: F401
    all_forms_config,
    standard_config,
    swept_fraction_config,
    write_yaml,
)
from tests.e2e.fixtures.paths import (  # noqa: F401
    output_root,
    ledger_file,
    samples_dir,
)
from tests.fixtures.packaging_fixtures import built_wheels_dir  # noqa: F401
from tests.utils.seeds import get_test_seed


@pytest.fixture(scope="session")
def test_seed() -> int:
    """Return the single seed that drives every fixture in the run.

    All randomized fixture builders (synthetic images, parquet sidecars,
    configs) are derived from this value, so a full ``pytest`` invocation is
    byte-reproducible for a given ``KCAI_TEST_SEED``.
    """
    return get_test_seed()