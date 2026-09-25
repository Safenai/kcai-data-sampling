"""Opt-in registry contents once ``-lama`` is installed.

The default-env registry tests in ``tests/unit/core/test_registry.py`` are
env-agnostic by design (superset / module-root assertions). These pin the
opt-in additions exactly: a models registry holding precisely the
``lama_inpaint`` plugin and a transformations registry that gains ``inpaint``,
both exported from the ``-lama`` module roots.
"""

import pytest

from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry, get_transformations_registry

pytest.importorskip("kcai_data_sampling_images_lama.api.models.lama")


def test_models_registry_holds_exactly_the_lama_inpaint_plugin() -> None:
    """With ``-lama`` installed, the models registry is not empty.

    The plugin is the only model adapter, registered under ``lama_inpaint`` and
    exported from the ``-lama`` package root.
    """
    registry = PluginLoadedRegistry.get_models_registry()
    assert set(registry) == {"lama_inpaint"}
    assert registry["lama_inpaint"].__module__.startswith("kcai_data_sampling_images_lama")


def test_transformations_registry_gains_inpaint_from_the_lama_root() -> None:
    """``inpaint`` joins the transformations superset, exported from ``-lama``.

    The default trio stays put (the superset assertion in the default-env
    tests), and the new algorithm is the ``-lama`` one.
    """
    registry = get_transformations_registry()
    assert {"horizontal_flip", "crop_resize"} <= set(registry)
    assert registry["inpaint"].__module__.startswith("kcai_data_sampling_images_lama")