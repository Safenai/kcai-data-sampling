"""Deterministic seeding for the kcai test suite.

The whole suite is reproducible: every run draws from a single seed that can
be overridden through the ``KCAI_TEST_SEED`` environment variable (the nox
sessions set it to 42). The default lives here so `get_test_seed()` returns a
sane value even under a bare `pytest` invocation with no environment set.
"""

import os

KCAI_TEST_SEED = 42


def get_test_seed() -> int:
    """Return the suite seed for the current run.

    Reads ``KCAI_TEST_SEED`` from the environment (as set by the ``test`` and
    ``test_packaging`` nox sessions) and falls back to `KCAI_TEST_SEED` when
    the variable is unset, so fixtures seeded from the return value stay
    byte-reproducible across CI, nox, and local runs.
    """
    return int(os.environ.get("KCAI_TEST_SEED", str(KCAI_TEST_SEED)))