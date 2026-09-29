"""Plugin discovery: entry-point iteration, model-shape probing, and path loads.

``load_registered_plugins`` filters by base class and skips the base name;
``_model_shaped`` probes the protocol shape (a non-empty ``name`` plus a model
method); ``load_model_source`` refuses a file it cannot build a spec for.
"""

import importlib.metadata
import logging
import types

from kcai_data_sampling_core.utils.registry import _model_shaped, load_model_source, load_registered_plugins
import pytest


class _MarkerBase:
    pass


class _MarkerSubclass(_MarkerBase):
    pass


class _MarkerForeign:
    pass


class _FakeEntryPoint:
    def __init__(self, name: str, obj: object) -> None:
        self.name = name
        self.obj = obj

    def load(self) -> object:
        return self.obj


def test_load_registered_plugins_filters_by_base_class_and_base_name(monkeypatch) -> None:
    """The ``default`` name is skipped; non-subclasses are logged and dropped."""
    entries = [
        _FakeEntryPoint("default", _MarkerBase),
        _FakeEntryPoint("sub", _MarkerSubclass),
        _FakeEntryPoint("foreign", _MarkerForeign),
    ]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group=None: entries)
    registry = load_registered_plugins("kcai_data_sampling.transformations", _MarkerBase)
    assert registry == {"sub": _MarkerSubclass}


def test_load_registered_plugins_without_a_base_class_keeps_everything(monkeypatch) -> None:
    """``base_class=None`` disables the type check."""
    entries = [_FakeEntryPoint("sub", _MarkerSubclass), _FakeEntryPoint("foreign", _MarkerForeign)]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group=None: entries)
    registry = load_registered_plugins("kcai_data_sampling.models", None)
    assert registry == {"sub": _MarkerSubclass, "foreign": _MarkerForeign}


def test_load_registered_plugins_tolerates_a_typeerror_from_entry_points(monkeypatch, caplog) -> None:
    """A pre-3.10 ``entry_points`` refusing ``group=`` yields an empty registry."""

    def broken(group=None):
        raise TypeError("entry_points() got an unexpected keyword argument 'group'")

    monkeypatch.setattr(importlib.metadata, "entry_points", broken)
    with caplog.at_level(logging.WARNING):
        assert load_registered_plugins("g", _MarkerBase) == {}
    assert "Old python version not supported" in caplog.text


class _Modelled:
    name = "modelled"

    def grad(self, xs):
        return xs


class _NamedOnly:
    name = "named"


def test_model_shaped_rejects_a_module_and_name_less_or_method_less_symbols() -> None:
    """Modules, anonymous objects, and empty names do not look like adapters."""
    assert _model_shaped(types.ModuleType("mod")) is False
    assert _model_shaped(object()) is False
    assert _model_shaped(_NamedOnly) is False


def test_model_shaped_accepts_a_name_plus_a_model_method() -> None:
    """A non-empty ``name`` plus ``grad`` (or ``inpaint``) reads as an adapter."""
    assert _model_shaped(_Modelled) is True


def test_load_model_source_refuses_a_file_without_a_spec(tmp_path, monkeypatch) -> None:
    """A ``None`` spec from the file location is a load refusal, not a crash."""
    source = tmp_path / "model.py"
    source.write_text("x = 1\n")
    monkeypatch.setattr("importlib.util.spec_from_file_location", lambda name, path: None)
    with pytest.raises(ValueError, match=r"cannot load .*model\.py as a python module"):
        load_model_source("m", path=str(source))
