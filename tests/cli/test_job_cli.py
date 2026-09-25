"""The job CLI (``kcai-data-sampling-job``): argument handling and guards.

The surface: ``parse_args`` understands ``-p``/``--process-config``;
``execute`` runs each YAML with a mocked ``run`` and honors ``--save-config``;
a YAML parse error or a missing config file is tolerated (logged, returned),
not raised; ``run`` itself guards the empty config and the registry-resolved
dataloader/transformation types with the registered ``ValueError``s.
"""

from pathlib import Path

import pytest
import yaml

from kcai_data_sampling_job import cli as job_cli
from kcai_data_sampling_job.cli import execute, parse_args, run
from tests.e2e.fixtures.configs import write_yaml
from tests.utils.configs import build_config, build_loader


def test_parse_args_accepts_short_and_long_config_flags() -> None:
    """``-p`` and ``--process-config`` both populate ``process_config``.

    The short form takes several files; the long form mirrors it — the two
    spellings a user meets in the README behave identically.
    """
    short = parse_args(["-p", "a.yaml", "b.yaml"])
    assert short.process_config == ["a.yaml", "b.yaml"]
    long = parse_args(["--process-config", "a.yaml", "b.yaml"])
    assert long.process_config == ["a.yaml", "b.yaml"]


def test_parse_args_requires_a_config() -> None:
    """A call without ``-p`` terminates with the parser's usage error."""
    with pytest.raises(SystemExit):
        parse_args([])


def test_execute_runs_the_config_with_a_mocked_run(monkeypatch, tmp_path: Path) -> None:
    """``execute(["-p", valid])`` loads the YAML and hands it to ``run``.

    The config dict the mocked ``run`` receives is exactly the YAML that was
    written — the CLI's only job is to turn the file into a dict.
    """
    config_path = write_yaml(tmp_path / "cfg.yaml", {"compute": {"progress_bar": False}})
    received: dict = {}

    def fake_run(config: dict):
        received.update(config)

    monkeypatch.setattr(job_cli, "run", fake_run)

    execute(["-p", str(config_path)])
    assert received == {"compute": {"progress_bar": False}}


def test_execute_save_config_writes_the_resolved_config(monkeypatch, tmp_path: Path) -> None:
    """``--save-config`` dumps the parsed config to disk untouched.

    The resolved configuration file (what a later ``-p`` would consume) is
    written even though ``run`` is mocked away.
    """
    config_path = write_yaml(tmp_path / "cfg.yaml", {"compute": {"progress_bar": False}})
    save_path = tmp_path / "out" / "resolved.yaml"

    monkeypatch.setattr(job_cli, "run", lambda config: None)
    execute(["-p", str(config_path), "--save-config", str(save_path)])

    assert yaml.safe_load(save_path.read_text()) == {"compute": {"progress_bar": False}}


def test_execute_tolerates_a_missing_config_file(monkeypatch, tmp_path: Path) -> None:
    """A nonexistent config file is logged and skipped, never raised.

    dqm-ml's graceful shape: the CLI reports and returns so a user's typo does
    not explode mid-command.
    """
    calls: list[dict] = []
    monkeypatch.setattr(job_cli, "run", lambda config: calls.append(config))

    execute(["-p", str(tmp_path / "absent.yaml")])
    assert calls == []


def test_execute_tolerates_a_yaml_parse_error(monkeypatch, tmp_path: Path) -> None:
    """A malformed YAML is logged and returns without calling ``run``.

    An unclosed bracket parses as a ``YAMLError``; the CLI catches it, prints
    the error, and stops — no ``run``, no raise.
    """
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("[[[]\n  still_open\n$%^@#\n")
    calls: list[dict] = []
    monkeypatch.setattr(job_cli, "run", lambda config: calls.append(config))

    execute(["-p", str(bad_path)])
    assert calls == []


def test_run_guards_the_empty_config() -> None:
    """``run({})`` raises kcai's own ValueError, before any validation.

    The guard is ours (not pydantic's): an empty dict is refused loudly.
    """
    with pytest.raises(ValueError, match="requires a configuration"):
        run({})


def test_run_refuses_an_unknown_transformation_type(tmp_path: Path) -> None:
    """An unregistered transformation ``type`` fails with the registered error."""
    config = build_config(
        loaders=[build_loader(parquet_path="missing.parquet")],
        output_path=str(tmp_path / "ledger.parquet"),
        samples_dir=str(tmp_path / "s"),
        transformations=[{"type": "no_such_algorithm"}],
    )
    with pytest.raises(ValueError, match="unknown transformation type 'no_such_algorithm'"):
        run(config)


def test_run_refuses_an_unknown_dataloader_type(tmp_path: Path) -> None:
    """An unregistered dataloader ``type`` fails with the registered error."""
    loader = build_loader(parquet_path="missing.parquet")
    loader["type"] = "csv"
    config = build_config(
        loaders=[loader],
        output_path=str(tmp_path / "ledger.parquet"),
        samples_dir=str(tmp_path / "s"),
    )
    with pytest.raises(ValueError, match="unknown dataloader type 'csv'"):
        run(config)