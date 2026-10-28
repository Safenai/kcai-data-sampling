"""Code generator: specs → configs.py + api/transformations/*.py + entry points.

Run as ``python -m kcai_data_sampling_albumentation._generate`` to regenerate
after editing ``specs.py``; ``--check`` (used by the freshness test) renders
without writing and fails if any committed file is stale. Everything generated
is committed — no build-time magic.
"""

from pathlib import Path
import re
import sys
from typing import Any

from kcai_data_sampling_albumentation.specs import TRANSFORM_SPECS, AlbuSpec, Param

_PACKAGE_ROOT = Path(__file__).resolve().parent
_CONFIGS_PATH = _PACKAGE_ROOT / "configs.py"
_API_DIR = _PACKAGE_ROOT / "api" / "transformations"
_PYPROJECT_PATH = _PACKAGE_ROOT.parent.parent / "pyproject.toml"

#: ruff's configured line length ([tool.ruff] line-length in pyproject.toml).
#: The generator must wrap a collection the same way ruff format would, or the
#: committed file would drift the first time `nox -s fmt` runs.
_LINE_LENGTH = 120

#: Splits a name into alternating text / digit runs for the natural sort
#: (RUF022's ``__all__`` ordering): ``D10`` → ``["D", "10", ""]``.
_NATURAL_SPLIT = re.compile(r"(\d+)")

_GENERATED_HEADER = (
    '"""GENERATED from specs.py by kcai_data_sampling_albumentation._generate — do not edit.\n'
    "\n"
    "One config schema / transformation class per allowlisted upstream transform.\n"
    "Regenerate with `python -m kcai_data_sampling_albumentation._generate`.\n"
    '"""\n'
)

_ENTRY_BEGIN = "# BEGIN generated entry points (kcai_data_sampling_albumentation._generate)"
_ENTRY_END = "# END generated entry points"


def _pascal(name: str) -> str:
    """Convert a snake_case type name to its PascalCase class name.

    Args:
        name: The registered type (``random_crop``).

    Returns:
        The class name (``RandomCrop``).
    """
    return "".join(part.capitalize() for part in name.split("_"))


def _dunder_all_key(name: str) -> tuple[int, list[Any]]:
    """Sort key matching ruff RUF022 (isort-style) ordering of ``__all__``.

    Ruff orders re-exported names isort-style: all-uppercase "constants"
    (``CLAHE``, ``D4``) first, then CamelCase classes, then lowercase — a
    natural sort within each group (digit runs compare as integers, so ``D4``
    precedes ``D10``). Rendering ``__all__`` in this order is what keeps the
    generated file RUF022-clean by construction, since ``nox -s fmt`` runs
    isort but never RUF022's fixer.

    Args:
        name: One exported name.

    Returns:
        A ``(category, natural_key)`` tuple; lower is earlier.
    """
    category = 0 if name.isupper() and any(c.isalpha() for c in name) else 1
    natural = [int(part) if part.isdigit() else part for part in _NATURAL_SPLIT.split(name) if part != ""]
    return (category, natural)


def _literal(value: Any) -> str:
    """Render a value as a double-quoted Python literal, as ruff formats it.

    ``repr`` emits single quotes for strings, which ruff format rewrites to
    double quotes — so generated text would never match its formatted form.
    Rendering double quotes here is what makes the output format-stable by
    construction. Non-strings pass through ``repr`` unchanged.

    Args:
        value: The value to render.

    Returns:
        The literal source.
    """
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def _default_literal(value: Any, kind: str) -> str:
    """Render a spec default as a Python literal for the generated schema.

    Args:
        value: The default from the spec table.
        kind: The parameter kind.

    Returns:
        The literal source (tuples become lists, per the schema annotation).
    """
    if value is None:
        return "None"
    if kind.startswith("range_pair") and isinstance(value, (list, tuple)):
        # Coerce the pair to the annotation's element type: a `range_pair_float`
        # whose spec default is written as ints (`(-90, 90)`) must render as
        # floats, or mypy rejects `list[int]` against `list[float]`.
        cast = float if kind == "range_pair_float" else int
        return f"[{cast(value[0])!r}, {cast(value[1])!r}]"
    if kind.startswith("range_pair"):
        # A scalar default for a pair-kind parameter (upstream's `rotate=0.0`):
        # the annotation admits a bare scalar, so render it as one.
        cast = float if kind == "range_pair_float" else int
        return repr(cast(value))
    return _literal(value)


def _field_annotation(param: Param) -> str:
    """Render the schema annotation for one parameter kind.

    Args:
        param: The spec row's parameter.

    Returns:
        The annotation source: pairs accept a fixed scalar too (a sweep
        expands to scalars, and albumentations broadcasts them symmetrically).
    """
    if param.kind == "range_pair_float":
        base = "list[float] | float | SweepConfig"
    elif param.kind == "range_pair_int":
        base = "list[int] | int | SweepConfig"
    elif param.kind == "enum":
        choices = ", ".join(_literal(c) for c in (param.choices or {}))
        base = f"Literal[{choices}]"
    elif param.kind == "bool":
        base = "bool"
    elif param.kind == "fill":
        base = "int"
    elif param.kind == "size":
        base = "int | SweepConfig"
    else:
        base = "float | int | SweepConfig"  # scalar
    # Allow None for params whose upstream default is None ("auto-calculate").
    if param.default is None:
        base = f"{base} | None"
    return base


def _bounds_kwargs(param: Param) -> str:
    """Render the bound keyword arguments for one parameter.

    Args:
        param: The spec row's parameter.

    Returns:
        The keyword-argument source, e.g. ``", minimum=0.0"`` ("" when open).
    """
    minimum, maximum = param.bounds
    parts = []
    if minimum is not None:
        parts.append(f"minimum={minimum!r}")
    if maximum is not None:
        parts.append(f"maximum={maximum!r}")
    return (", " + ", ".join(parts)) if parts else ""


def _field_block(name: str, param: Param) -> str:
    """Render one schema field with its docstring-style description.

    Args:
        name: The upstream parameter name.
        param: The spec row's parameter.

    Returns:
        The field source block.
    """
    annotation = _field_annotation(param)
    # ``...`` (Ellipsis) marks an upstream-required parameter (``random_crop``'s
    # ``height``/``width``): the schema leaves it required, so omit ``default=``.
    default_line = "" if param.default is ... else f"        default={_default_literal(param.default, param.kind)},\n"
    return (
        f"    {name}: {annotation} = Field(\n"
        f"{default_line}"
        f'        description="{name} ({param.kind}); a SweepConfig expands it.",\n'
        f"    )\n"
    )


def _render_validator(type_name: str, params: dict[str, Param]) -> str:
    """Render the per-schema model validator enforcing every parameter bound.

    Args:
        type_name: The registered type (for the docstring).
        params: The spec row's parameters.

    Returns:
        The validator source block, including the ``p`` bounds check.
    """
    lines = [
        '    @model_validator(mode="after")',
        "    def _parameter_bounds(self) -> Self:",
        f'        """Enforce the algorithm bounds on {type_name} parameters, value or sweep."""',
    ]
    for name, param in params.items():
        if param.kind.startswith("range_pair"):
            lines.append(f'        self._check_range_pair("{name}"{_bounds_kwargs(param)})')
        elif param.kind in {"scalar", "size"}:
            lines.append(f'        self._check_parameter_bounds("{name}"{_bounds_kwargs(param)})')
    lines.append("        self._check_p()")
    lines.append("        return self")
    return "\n".join(lines) + "\n"


def _render_config(type_name: str, spec: AlbuSpec) -> str:
    """Render one generated config schema class.

    Args:
        type_name: The registered type (``gaussian_blur``).
        spec: Its spec row.

    Returns:
        The class source block.
    """
    cls = _pascal(type_name) + "TransformationConfig"
    doc = f'    """Configuration of the ``{type_name}`` transformation."""\n'
    body = [f"\n\nclass {cls}(AlbumentationsConfig):\n", doc, f'\n    type: Literal["{type_name}"] = "{type_name}"\n']
    for name, param in spec.params.items():
        body.append(_field_block(name, param))
    body.append("\n")
    body.append(_render_validator(type_name, spec.params))
    return "".join(body)


def _render_configs() -> str:
    """Render the full generated ``configs.py``.

    Returns:
        The file source.
    """
    header = [
        _GENERATED_HEADER,
        "\nfrom typing import Literal, Self\n\n",
        "from kcai_data_sampling_core.models.sweep import SweepConfig\n",
        "from pydantic import Field, model_validator\n",
        "\nfrom kcai_data_sampling_albumentation.config_base import AlbumentationsConfig\n",
    ]
    return "".join(header) + "".join(_render_config(name, spec) for name, spec in TRANSFORM_SPECS.items())


def _render_class_doc(type_name: str, spec: AlbuSpec) -> str:
    """Render the docstring of one generated transformation class.

    Args:
        type_name: The registered type.
        spec: Its spec row.

    Returns:
        The docstring source.
    """
    params = ", ".join(spec.params) or "none"
    return (
        f'    """The ``{type_name}`` transformation ({spec.family} family).\n'
        f"\n"
        f"    Wraps upstream ``{spec.source}`` via the shared engine: per-row re-seeding,\n"
        f"    alpha-split for image-only transforms, uint8 shape contract.\n"
        f"    Exposed parameters: {params}.\n"
        f'    """\n'
    )


def _render_class(type_name: str, spec: AlbuSpec) -> str:
    """Render one generated transformation class module.

    Args:
        type_name: The registered type.
        spec: Its spec row.

    Returns:
        The module source.
    """
    cls = _pascal(type_name)
    config_cls = cls + "TransformationConfig"
    lines = [
        _GENERATED_HEADER,
        "\nimport albumentations as A  # noqa: N812\n\n",
        f"from kcai_data_sampling_albumentation.configs import {config_cls}\n",
        "from kcai_data_sampling_albumentation.engine import AlbumentationsUnary\n\n\n",
        f"class {cls}(AlbumentationsUnary):\n",
        _render_class_doc(type_name, spec),
        f'\n    algorithm = "{type_name}"\n\n',
        f"    Config = {config_cls}\n\n",
        f"    transform_class = A.{spec.source}\n",
    ]
    if spec.geometry is not None:
        lines.append(f'\n    geometry = "{spec.geometry}"\n')
    if spec.pinned:
        pinned = ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in spec.pinned.items())
        lines.append(f"\n    pinned = {{{pinned}}}\n")
    lines.append(_render_enum_maps(spec))
    return "".join(lines)


def _render_enum_maps(spec: AlbuSpec) -> str:
    """Render the ``enum_maps`` class attribute for one spec row.

    Always multiline, one parameter per line with a trailing comma, and a
    nested mapping wrapped when its single-line form would exceed the
    configured line length: this is exactly the shape ruff format leaves
    untouched, which is what keeps the generated file format-stable.

    Args:
        spec: The spec row.

    Returns:
        The ``enum_maps`` block source, or the empty string when there are no
        enum parameters.
    """
    enum_maps = {name: p.choices for name, p in spec.params.items() if p.kind == "enum" and p.choices}
    if not enum_maps:
        return ""
    lines = ["\n    enum_maps = {\n"]
    for name, mapping in enum_maps.items():
        inner = ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in mapping.items())
        one_line = f"        {_literal(name)}: {{{inner}}},\n"
        if len(one_line.rstrip("\n")) <= _LINE_LENGTH:
            lines.append(one_line)
        else:
            lines.append(f"        {_literal(name)}: {{\n")
            for key, value in mapping.items():
                lines.append(f"            {_literal(key)}: {_literal(value)},\n")
            lines.append("        },\n")
    lines.append("    }\n")
    return "".join(lines)


def _render_api_init() -> str:
    """Render the generated ``api/transformations/__init__.py`` re-exports.

    Returns:
        The module source.
    """
    lines = [_GENERATED_HEADER, "\n"]
    # isort (the `fmt` session) sorts imports by module path; render them in
    # that order so the generated file is already isort-stable.
    for type_name in sorted(TRANSFORM_SPECS):
        cls = _pascal(type_name)
        lines.append(f"from kcai_data_sampling_albumentation.api.transformations.{type_name} import {cls}\n")
    # RUF022 (not autofixed by `fmt`) wants ``__all__`` sorted isort-style
    # (all-caps constants first, then CamelCase, natural-sorted); render it that
    # way and multiline with a magic trailing comma, which ruff format preserves.
    lines.append("\n__all__ = [\n")
    for cls in sorted((_pascal(name) for name in TRANSFORM_SPECS), key=_dunder_all_key):
        lines.append(f'    "{cls}",\n')
    lines.append("]\n")
    return "".join(lines)


def _render_entry_points() -> str:
    """Render the ``kcai_data_sampling.transformations`` entry-point block.

    Returns:
        The TOML block, one line per allowlisted type.
    """
    return "\n".join(
        f'{name} = "kcai_data_sampling_albumentation.api.transformations.{name}:{_pascal(name)}"'
        for name in TRANSFORM_SPECS
    )


def _replace_entry_block(pyproject: str) -> str:
    """Replace the marker-delimited entry-point block in pyproject.toml.

    Args:
        pyproject: The current pyproject.toml source.

    Returns:
        The source with the entry points regenerated from the spec table.

    Raises:
        ValueError: If the marker pair is missing.
    """
    begin = pyproject.find(_ENTRY_BEGIN)
    end = pyproject.find(_ENTRY_END)
    if begin == -1 or end == -1:
        raise ValueError("pyproject.toml: entry-point markers not found")
    head = pyproject[: begin + len(_ENTRY_BEGIN)]
    tail = pyproject[end:]
    return f"{head}\n{_render_entry_points()}\n{tail}"


def render_all() -> dict[Path, str]:
    """Render every generated artifact without writing it.

    Returns:
        Path → expected source, including pyproject.toml.
    """
    return {
        _CONFIGS_PATH: _render_configs(),
        _API_DIR / "__init__.py": _render_api_init(),
        **{_API_DIR / f"{name}.py": _render_class(name, spec) for name, spec in TRANSFORM_SPECS.items()},
        _PYPROJECT_PATH: _replace_entry_block(_PYPROJECT_PATH.read_text()),
    }


def orphans() -> list[Path]:
    """Generated-module paths on disk that no spec row produces any more.

    A row removed from the table (or a spec that fails the contract and is
    dropped) leaves its ``api/transformations/<name>.py`` behind, importing a
    config class that no longer exists — mypy and the imports catch it, but the
    generator should prune it. ``--check`` treats an orphan as stale.

    Returns:
        The orphaned module paths, sorted.
    """
    expected = {_API_DIR / f"{name}.py" for name in TRANSFORM_SPECS} | {_API_DIR / "__init__.py"}
    return sorted(path for path in _API_DIR.glob("*.py") if path not in expected)


def _apply_orphans(check: bool, stale: list[Path]) -> None:
    """Prune orphaned generated modules, or record them as stale under ``--check``.

    Args:
        check: Whether ``main`` runs in verify mode.
        stale: The stale-path list to extend in verify mode (mutated).
    """
    orphaned = orphans()
    if check:
        stale.extend(orphaned)
        return
    for path in orphaned:
        path.unlink()
        print(f"removed orphan {path.relative_to(_PACKAGE_ROOT.parent.parent)}")


def main(argv: list[str] | None = None) -> int:
    """Write (or with ``--check``, verify) every generated artifact.

    Args:
        argv: Command-line arguments; ``--check`` verifies freshness instead
            of writing.

    Returns:
        Process exit code: 0 when fresh/written, 1 when ``--check`` finds a
        stale artifact.
    """
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    stale: list[Path] = []
    for path, content in render_all().items():
        if check:
            if not path.is_file() or path.read_text() != content:
                stale.append(path)
        else:
            path.write_text(content)
            print(f"wrote {path.relative_to(_PACKAGE_ROOT.parent.parent)}")
    _apply_orphans(check, stale)
    if stale:
        for path in stale:
            print(f"stale: {path}")
        print("run: python -m kcai_data_sampling_albumentation._generate")
        return 1
    if check:
        print("generated files are fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
