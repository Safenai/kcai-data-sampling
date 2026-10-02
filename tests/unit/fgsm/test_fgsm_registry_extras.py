"""Opt-in registry contents once ``-fgsm`` is installed.

The default-env registry tests in ``tests/unit/core/test_registry.py`` are
env-agnostic by design (superset / module-root assertions). These pin the
opt-in adversarial additions exactly: the transformations registry gains
``fgsm`` exported from the ``-fgsm`` package root, and no model adapter ships
with the adversarial package — the target is the user's, provided at run time.
"""

from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry, get_transformations_registry
import pytest

pytest.importorskip("kcai_data_sampling_fgsm")

pytestmark = pytest.mark.fgsm


def test_transformations_registry_gains_fgsm_from_its_package_root() -> None:
    """``fgsm`` joins the transformations superset, exported from ``-fgsm``.

    The default trio stays put (the superset assertion in the default-env
    tests), and the new algorithm is the ``-fgsm`` one, not a core class.
    """
    registry = get_transformations_registry()
    assert {"horizontal_flip", "crop_resize"} <= set(registry)
    assert registry["fgsm"].__module__.startswith("kcai_data_sampling_fgsm")


def test_the_adversarial_package_registers_no_models() -> None:
    """``-fgsm`` ships no model adapter: no model class is exported from its root.

    The adversarial package computes *against* the user's target model — the
    isolation claimed in the default-env registry tests — so nothing in the
    models registry carries an ``-fgsm`` module no matter which opt-in
    packages happen to be co-installed in the same environment.
    """
    registry = PluginLoadedRegistry.get_models_registry()
    assert not any(cls.__module__.startswith("kcai_data_sampling_fgsm") for cls in registry.values())
