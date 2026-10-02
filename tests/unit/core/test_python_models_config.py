"""The ``type: python`` model reference at config-load time.

The bring-your-own-code route is a **load-time** contract: ``ModelRefConfig``
enforces the reserved ``python`` kind's own field rules (exactly one of
``path``/``module``; source fields are refused on a registered plugin type),
and ``ModelsConfig`` resolves every ``type: python`` source when the config
loads — a missing file fails the load, and executing the referenced code by
design is visible at load, not mid-run.
"""

from __future__ import annotations

from pathlib import Path

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_core.models.models import ModelRefConfig, ModelsConfig
import pydantic
import pytest

from tests.fixtures.registries import StubTool, set_models_registry
from tests.utils.configs import build_config, build_loader

_ADAPTER_SOURCE = '''\
"""A stateless target-role stand-in adapter for the python-kind tests."""

class StubP:
    """A grad-carrying adapter, named like the ``models:`` key it serves."""

    name = "stub_p"

    def __init__(self, weights=None, **params):
        self.weights = weights
        self.params = params

    def grad(self, xs):
        s.setdefault("calls", 0)
        s["calls"] += 1
        return xs * 0.0
'''


@pytest.fixture
def adapter_file(tmp_path: Path) -> Path:
    """A writable ``type: python`` adapter file for one test."""
    path = tmp_path / "stub_p.py"
    path.write_text(_ADAPTER_SOURCE, encoding="utf-8")
    return path


def test_python_kind_requires_exactly_one_of_path_or_module() -> None:
    """``type: python`` needs exactly one source: ``path`` xor ``module``.

    Both missing and both present are config errors; the kind is reserved, so
    it never falls through to the registry.
    """
    with pytest.raises(pydantic.ValidationError, match="exactly one"):
        ModelRefConfig(type="python")
    with pytest.raises(pydantic.ValidationError, match="exactly one"):
        ModelRefConfig(type="python", path="a.py", module="pkg.a")


def test_plugin_type_refuses_python_source_fields(monkeypatch) -> None:
    """A registered plugin type refuses the ``python`` source fields.

    ``path``/``module``/``export`` are only meaningful for the reserved kind;
    carrying them on a plugin reference is a config error naming the field.
    """
    set_models_registry(monkeypatch, {"stub": StubTool})
    with pytest.raises(pydantic.ValidationError) as exc:
        ModelRefConfig(type="stub", path="x.py")
    assert "is a registered plugin" in str(exc.value)
    assert "'path' is only valid for type: python" in str(exc.value)


def test_job_config_resolves_a_python_reference_at_load(tmp_path: Path, adapter_file: Path) -> None:
    """A job config with a ``type: python`` reference loads its source.

    ``JobConfig.model_validate`` reads the adapter file by design: validation
    succeeds, the reference keeps its python kind, and the file was executed
    exactly once at load (the mark is written by the module body).
    """
    mark = tmp_path / "loaded.mark"

    marked_source = _ADAPTER_SOURCE + f"\nmark = {str(mark)!r}\nopen(mark, 'w').close()\n"
    adapter_file.write_text(marked_source, encoding="utf-8")
    config = build_config(
        loaders=[build_loader(parquet_path="missing.parquet")],
        output_path=str(tmp_path / "ledger.parquet"),
        samples_dir=str(tmp_path / "s"),
        models={"stub_p": {"type": "python", "path": str(adapter_file)}},
    )
    validated = JobConfig.model_validate(config)
    assert validated.models.models["stub_p"].type == "python"
    assert mark.is_file()


def test_missing_python_source_fails_config_load(tmp_path: Path) -> None:
    """A missing ``type: python`` source fails the load, not the run.

    ``ModelsConfig`` resolves the reference up front, so the loud "file not
    found" error surfaces at ``model_validate`` time.
    """
    config = build_config(
        loaders=[build_loader(parquet_path="missing.parquet")],
        output_path=str(tmp_path / "ledger.parquet"),
        samples_dir=str(tmp_path / "s"),
        models={"m": {"type": "python", "path": str(tmp_path / "absent.py")}},
    )
    with pytest.raises(pydantic.ValidationError) as exc:
        JobConfig.model_validate(config)
    assert "file not found" in str(exc.value)


def test_models_config_preserves_the_python_reference_fields(adapter_file: Path) -> None:
    """A python reference keeps its source fields after validation.

    ``path``/``module``/``export`` survive ``ModelsConfig`` validation so the
    loader (and the config dump) still sees them; loading is side-effect-free
    beyond the intended execution.
    """
    section = ModelsConfig.model_validate({"stub_p": {"type": "python", "path": str(adapter_file), "export": "StubP"}})
    ref = section.models["stub_p"]
    assert (ref.type, ref.export) == ("python", "StubP")
    assert ref.path == str(adapter_file)
    assert ref.module is None
