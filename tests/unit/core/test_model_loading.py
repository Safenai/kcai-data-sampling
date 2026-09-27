"""``load_model_source``: the bring-your-own-code file and module routes.

The user adapter loader executes the referenced source by design and caches it
under a path-derived module name so a file is executed exactly once. This
module pins the resolution rules (absolute-first vs CWD-relative ``path``,
dotted ``module``), the ``export`` default ladder (explicit, key-named, sole
model-shaped symbol, loud refusals), the loud failure messages, and the
single-file limitation (no relative imports).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kcai_data_sampling_core.utils.registry import load_model_source


@pytest.fixture
def adapter_file(tmp_path: Path) -> Path:
    """A grad-carrying adapter class for the file-route tests."""
    path = tmp_path / "stub_adapter.py"
    path.write_text(
        '''\
class Adapter:
    """A grad-carrying stand-in adapter."""

    name = "stub_adapter"

    def __init__(self, weights=None, **params):
        self.weights = weights
        self.params = params

    def grad(self, xs):
        return xs * 0.0
''',
        encoding="utf-8",
    )
    return path


def test_python_source_requires_exactly_one_of_path_or_module() -> None:
    """Both kinds void demand exactly one of ``path``/``module``."""
    with pytest.raises(ValueError, match="needs exactly one"):
        load_model_source("m")
    with pytest.raises(ValueError, match="needs exactly one"):
        load_model_source("m", path="a.py", module="pkg.a")


def test_file_route_resolves_an_absolute_path_and_dotted_export(adapter_file: Path) -> None:
    """Absolute ``path`` + dotted ``export`` returns the nested symbol.

    ``export`` is a dotted attribute path into the loaded module; the returned
    symbol is the class the test then treats as the adapter.
    """
    assert load_model_source("adapter", path=str(adapter_file), export="Adapter").name == "stub_adapter"


def test_file_route_resolves_relative_to_the_working_directory(monkeypatch, tmp_path: Path) -> None:
    """A relative ``path`` resolves against the working directory.

    ``load_model_source`` tries the path as given (absolute first), else the
    process CWD — a ``monkeypatch.chdir`` into the staging dir is how a
    relative config path is exercised.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel_adapter.py").write_text(
        'class M:\n    name = "rel"\n    def grad(self, xs):\n        return xs * 0.0\n',
        encoding="utf-8",
    )
    loaded = load_model_source("m", path="rel_adapter.py")
    assert loaded.name == "rel"


def test_module_route_supports_relative_imports(tmp_path: Path) -> None:
    """A dotted ``module`` resolves a package and its relative imports.

    Unlike a single file, a package can import its siblings; the export is
    resolved from the loaded module exactly as for the file route.
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mixins.py").write_text("HELPER = 'ok'\n", encoding="utf-8")
    (tmp_path / "pkg" / "adapters.py").write_text(
        'from .mixins import HELPER\n\n'
        'class Adapter:\n'
        '    name = "pkg"\n'
        '    def grad(self, xs):\n'
        '        return xs * 0.0\n',
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        assert load_model_source("m", module="pkg.adapters", export="Adapter").name == "pkg"
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("pkg", None)
        sys.modules.pop("pkg.adapters", None)
        sys.modules.pop("pkg.mixins", None)


def test_module_route_import_failure_is_loud() -> None:
    """An unimportable ``module`` is wrapped into the loud refusal."""
    with pytest.raises(ValueError, match="cannot import module"):
        load_model_source("m", module="kcai_data_sampling_no_such_module_xyz")


def test_missing_file_is_loud(adapter_file: Path) -> None:
    """A missing ``path`` names the resolved file in the refusal."""
    missing = adapter_file.parent / "absent.py"
    with pytest.raises(ValueError) as exc:
        load_model_source("m", path=str(missing))
    assert "file not found" in str(exc.value)
    assert str(missing) in str(exc.value)


def test_exec_error_is_wrapped_as_raised_while_loading(tmp_path: Path) -> None:
    """A source that raises while executing surfaces as a load error."""
    bad = tmp_path / "broken.py"
    bad.write_text('raise RuntimeError("boom")\n', encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_model_source("m", path=str(bad))
    assert "raised while loading" in str(exc.value)
    assert "boom" in str(exc.value)


def test_export_defaults_to_the_key_named_symbol(tmp_path: Path) -> None:
    """Unset, ``export`` prefers the module attribute named like the key."""
    path = tmp_path / "yolo.py"
    path.write_text(
        'class yolo:\n'
        '    name = "key_named"\n'
        '    def grad(self, xs):\n'
        '        return xs * 0.0\n',
        encoding="utf-8",
    )
    assert load_model_source("yolo", path=str(path)).name == "key_named"


def test_export_defaults_to_the_sole_model_shaped_symbol(tmp_path: Path) -> None:
    """Unset, ``export`` falls back to the sole model-shaped symbol."""
    path = tmp_path / "only.py"
    path.write_text(
        'class TheOnly:\n'
        '    name = "sole"\n'
        '    def grad(self, xs):\n'
        '        return xs * 0.0\n',
        encoding="utf-8",
    )
    assert load_model_source("m", path=str(path)).name == "sole"


def test_ambiguous_exports_are_refused_loudly(tmp_path: Path) -> None:
    """Several model-shaped symbols without ``export`` list the candidates."""
    path = tmp_path / "many.py"
    path.write_text(
        'class A:\n'
        '    name = "a"\n'
        '    def grad(self, xs):\n'
        '        return xs * 0.0\n'
        'class B:\n'
        '    name = "b"\n'
        '    def grad(self, xs):\n'
        '        return xs * 0.0\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_model_source("m", path=str(path))
    assert "exposes several model candidates" in str(exc.value)
    assert "A" in str(exc.value) and "B" in str(exc.value)


def test_no_model_candidate_is_refused_loudly(tmp_path: Path) -> None:
    """A source without a model-shaped symbol names its public symbols."""
    path = tmp_path / "empty.py"
    path.write_text('VALUE = 42\n', encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_model_source("m", path=str(path))
    assert "exposes no model candidate" in str(exc.value)
    assert "VALUE" in str(exc.value)


def test_bad_explicit_export_is_refused_loudly(adapter_file: Path) -> None:
    """An ``export`` naming nothing is refused with the module origin."""
    with pytest.raises(ValueError) as exc:
        load_model_source("m", path=str(adapter_file), export="Missing")
    assert "export 'Missing' not found in" in str(exc.value)


def test_a_file_is_executed_exactly_once(tmp_path: Path) -> None:
    """The path-derived module cache runs a file once, however often loaded.

    Two loads of the same path return the same cached module; the module body
    (which appends to a marker file) runs exactly once.
    """
    path = tmp_path / "once.py"
    path.write_text(
        'from pathlib import Path\n'
        'with (Path(__file__).parent / "calls.txt").open("a") as fh:\n'
        '    fh.write("called\\n")\n'
        'class S:\n'
        '    name = "once"\n'
        '    def grad(self, xs):\n'
        '        return xs * 0.0\n',
        encoding="utf-8",
    )
    assert load_model_source("s", path=str(path)).name == "once"
    assert load_model_source("s", path=str(path)).name == "once"
    assert (tmp_path / "calls.txt").read_text(encoding="utf-8") == "called\n"


def test_a_single_file_cannot_import_its_siblings(tmp_path: Path) -> None:
    """A standalone ``path`` file has no package context for relative modules.

    A sibling module sitting beside the file is *not* importable from it (only
    a dotted ``module`` package gets relative imports); the import failure is
    wrapped into the loud load error.
    """
    (tmp_path / "sibling.py").write_text('SIB = 1\n', encoding="utf-8")
    path = tmp_path / "main_file.py"
    path.write_text(
        'from sibling import SIB\n'
        'class S:\n'
        '    name = "main"\n'
        '    def grad(self, xs):\n'
        '        return xs * 0.0\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raised while loading"):
        load_model_source("s", path=str(path))