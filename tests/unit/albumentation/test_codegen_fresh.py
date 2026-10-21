"""Generated artifacts stay in sync with the curated spec table.

Everything under ``kcai_data_sampling_albumentation`` that is generated —
``configs.py``, the ``api/transformations`` modules, their ``__init__``
re-exports and the entry-point block in ``pyproject.toml`` — is committed and
rendered from ``specs.py`` by ``_generate.py``. This test renders the table
again in memory and refuses any drift, so a spec edit without a regeneration
fails here instead of shipping a stale schema.
"""

import pytest

pytest.importorskip("kcai_data_sampling_albumentation")

pytestmark = pytest.mark.albumentation

from kcai_data_sampling_albumentation._generate import orphans, render_all  # noqa: E402


def test_generated_artifacts_are_fresh() -> None:
    """Every committed generated artifact matches the spec table exactly.

    ``render_all`` includes ``pyproject.toml``: its entry-point block is
    regenerated from the same table, so adding a spec without re-running the
    generator leaves the entry points stale and is caught here too.
    """
    stale = [str(path) for path, content in render_all().items() if path.read_text() != content]

    assert not stale, (
        f"stale generated files: {stale}; "
        "run `python -m kcai_data_sampling_albumentation._generate` and commit the result"
    )


def test_no_orphaned_generated_modules() -> None:
    """No generated module outlives its spec row.

    Dropping a row (or a spec that fails the contract) must remove its
    ``api/transformations/<name>.py``; a leftover imports a config class that
    no longer exists, so the generator prunes it and ``--check`` flags it.
    """
    leftover = [str(path) for path in orphans()]

    assert not leftover, (
        f"orphaned generated modules: {leftover}; "
        "run `python -m kcai_data_sampling_albumentation._generate` to prune them"
    )
