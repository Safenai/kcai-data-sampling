"""``register_model``: in-process adapters, entry-point names, purity.

``register_model`` seeds the live models registry so a ``models:`` reference
can name an adapter built in the kernel (notebooks, gate smokes, tests)
without a plugin package. It is purely additive and never masks an entry-point
plugin — a claimed name is refused loudly — and because the mutation is never
reverted by the function, the tests pin the resulting config/build behavior and
hold the registry pure via the snapshot/restore fixture.
"""

from __future__ import annotations

import pytest

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_core.models.models import ModelRefConfig
from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry, register_model
from kcai_data_sampling_job.cli import _build_models
from tests.fixtures.registries import StubTarget, StubTool, set_models_registry
from tests.utils.configs import build_config, build_loader


def test_register_model_seeds_the_registry_for_config_and_build(registry_snapshot, tmp_path) -> None:
    """A registered adapter becomes a valid ``models:`` type at load and build.

    After ``register_model("stub", StubTarget)`` the type validates through
    ``ModelRefConfig``, a job config naming it loads, and ``_build_models``
    constructs the adapter instance by the ordinary plugin path.
    """
    register_model("stub", StubTarget)
    assert PluginLoadedRegistry.get_models_registry()["stub"] is StubTarget

    assert ModelRefConfig(type="stub").type == "stub"
    config = build_config(
        loaders=[build_loader(parquet_path="missing.parquet")],
        output_path=str(tmp_path / "ledger.parquet"),
        samples_dir=str(tmp_path / "s"),
        models={"stub": {"type": "stub"}},
    )
    validated = JobConfig.model_validate(config)
    built = _build_models(validated)
    assert set(built) == {"stub"}
    assert isinstance(built["stub"], StubTarget)
    assert built["stub"].name == "stub"


def test_register_model_refuses_a_name_an_entry_point_claims(monkeypatch) -> None:
    """A name a plugin already owns is refused rather than silently shadowed.

    ``register_model`` reloads the plugin registry first, so the injected
    ``lama_inpaint`` entry point keeps its name; the loud refusal tells the
    caller to pick a different name and update the ``models:`` section.
    """
    set_models_registry(monkeypatch, {"lama_inpaint": StubTool})
    with pytest.raises(ValueError) as exc:
        register_model("lama_inpaint", StubTarget)
    assert "already claimed by an entry-point plugin" in str(exc.value)


def test_registered_name_does_not_leak_beyond_the_test(registry_snapshot) -> None:
    """The snapshot/restore keeps a registration test pure.

    Registering ``stub`` here is visible during the test; the fixture restores
    the pre-test registry on teardown, so later tests (e.g. the next one) see
    the same contents as before.
    """
    register_model("stub", StubTarget)
    assert "stub" in PluginLoadedRegistry.get_models_registry()


def test_registry_is_pure_after_registration_tests() -> None:
    """No registration leaked into this test's view of the registry.

    ``registry_snapshot`` restored ``_models_registry`` after the seeding test;
    the models registry here holds only the installed plugins (none in the
    default env).
    """
    registry = PluginLoadedRegistry.get_models_registry()
    assert "stub" not in registry
    assert all(cls.__module__.startswith("kcai_data_sampling_") for cls in registry.values())