"""The umbrella CLI (``kcai-data-sampling``) dispatch tests.

Port of dqm-ml's ``unit/v2/test_main.py`` + ``test_dependency.py``: the four
package versions print under ``version``, the registry groups under ``list``,
``process`` appears exactly when ``-job`` imports (asserted with a patched
import through ``optional_dependencies``), and the help action dispatches to
the requested command.
"""

from contextlib import contextmanager

import pytest

from kcai_data_sampling import dependency
from kcai_data_sampling.__main__ import execute, parse_args
from kcai_data_sampling.dependency import get_available_command, optional_dependencies

ALL_COMMANDS = ["version", "list", "process"]


def test_version_prints_all_four_package_versions(capsys) -> None:
    """``version`` prints each installed package's version line.

    The umbrella, core, job and images versions are all present and parseable
    semver strings — nothing silently missing.
    """
    execute(["version"])
    out = capsys.readouterr().out
    lines = dict(line.split(": ", 1) for line in out.splitlines())
    assert set(lines) == {"kcai-data-sampling", "core", "job", "images"}
    import re

    for version in lines.values():
        assert re.fullmatch(r"\d+\.\d+\.\d+.*", version) is not None


def test_list_prints_the_registered_plugin_groups(capsys) -> None:
    """``list`` names every registered transformation, loader and writer.

    All four registry groups print; the image algorithms and the parquet/bytes
    plugins resolve from their entry points, so their classes appear.
    """
    execute(["list"])
    out = capsys.readouterr().out
    assert "Available data transformations" in out
    assert "- horizontal_flip" in out and "- crop_resize" in out
    assert "Available data models" in out
    assert "Available data dataloaders" in out
    assert "- parquet" in out
    assert "Available data output_writers" in out
    assert "- images" in out


def test_process_command_is_present_when_job_imports() -> None:
    """With ``-job`` installed, ``process`` is a registered command."""
    commands = get_available_command()
    assert {"version", "list", "process"} <= set(commands)
    assert commands["process"] is not None


def test_process_command_vanishes_without_the_job_package(monkeypatch, capsys) -> None:
    """With ``-job`` unimportable, only ``version``/``list`` survive.

    The optional import is exercised with a patched ``__import__`` that fails
    on ``-job``; ``optional_dependencies(error="warn")`` prints the warning and
    the command list shrinks accordingly.
    """
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "kcai_data_sampling_job.cli":
            raise ImportError("No module named 'kcai_data_sampling_job.cli'", name="kcai_data_sampling_job.cli")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    commands = get_available_command()
    assert set(commands) == {"version", "list"}
    out = capsys.readouterr().out
    assert "Warning: missing optional dependency" in out


def test_parse_args_dispatches_the_command_and_remainder() -> None:
    """``parse_args`` splits the command from its trailing arguments."""
    args, remaining = parse_args(["version"], ALL_COMMANDS)
    assert args.command == "version"
    assert remaining == []

    args, remaining = parse_args(["process", "-p", "cfg.yaml"], ALL_COMMANDS)
    assert args.command == "process"
    assert remaining == ["-p", "cfg.yaml"]


def test_parse_args_rejects_an_unknown_command() -> None:
    """An unlisted command is refused by argparse's choices, not dispatched."""
    with pytest.raises(SystemExit):
        parse_args(["bogus"], ALL_COMMANDS)


def test_help_action_dispatches_to_the_requested_command(capsys) -> None:
    """``<command> -h`` routes through the command's own handler.

    ``version -h`` runs the version handler (it ignores args), so the four
    versions print — the action rather than the global help fires.
    """
    execute(["version", "-h"])
    out = capsys.readouterr().out
    assert "kcai-data-sampling:" in out
    assert "core:" in out


def test_help_without_a_command_prints_the_global_help(capsys) -> None:
    """A bare ``-h`` shows the umbrella help and exits cleanly."""
    with pytest.raises(SystemExit):
        execute(["-h"])
    assert "kcai data-sampling job client" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["ignore", "warn"])
def test_optional_dependencies_swallows_missing_imports(capsys, mode: str) -> None:
    """``ignore`` and ``warn`` both tolerate a missing optional dependency."""
    with optional_dependencies(mode):
        raise ImportError("missing 'x'", name="x")
    if mode == "warn":
        assert "Warning: missing optional dependency x" in capsys.readouterr().out


def test_optional_dependencies_raise_rethrows_the_import_error() -> None:
    """``raise`` lets the missing-import error propagate."""
    with pytest.raises(ImportError, match="missing 'x'"):
        with optional_dependencies("raise"):
            raise ImportError("missing 'x'", name="x")


def test_optional_dependencies_rejects_an_unknown_mode() -> None:
    """A mode outside ``{raise, warn, ignore}`` is refused at entry."""
    with pytest.raises(AssertionError):
        optional_dependencies("bogus").__enter__()