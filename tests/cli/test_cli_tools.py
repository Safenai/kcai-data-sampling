"""Umbrella CLI utilities: the colored formatter and the console entry point.

``CustomFormatter`` paints log records by level, ``init_log`` installs it on
the root stream handlers, and ``main`` is the thin console-script hop into the
umbrella dispatcher.
"""

import logging

from kcai_data_sampling.cli_tools import Bcolors, CustomFormatter, main


def _record(message: str, levelno: int = logging.INFO) -> logging.LogRecord:
    """Build a minimal log record for formatting."""
    return logging.LogRecord(
        name="test",
        level=levelno,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_custom_formatter_colors_records_by_level() -> None:
    """Each level gets its ANSI color; the message is always present."""
    formatter = CustomFormatter()
    assert Bcolors.FAIL in formatter.format(_record("boom", logging.ERROR))
    assert Bcolors.OKBLUE in formatter.format(_record("trace", logging.DEBUG))
    assert Bcolors.WARNING in formatter.format(_record("careful", logging.WARNING))
    assert (Bcolors.FAIL + Bcolors.BOLD) in formatter.format(_record("fatal", logging.CRITICAL))
    assert "boom" in formatter.format(_record("boom", logging.ERROR))


def test_init_log_installs_the_custom_formatter_on_stream_handlers() -> None:
    """``init_log`` swaps the root stream handler formatter for ``CustomFormatter``."""
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)
    try:
        CustomFormatter.init_log(level=logging.INFO)
        assert isinstance(handler.formatter, CustomFormatter)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_main_dispatches_to_the_umbrella_cli(monkeypatch, capsys) -> None:
    """``main`` hops into ``execute`` with the process argv."""
    monkeypatch.setattr("sys.argv", ["kcai-data-sampling", "version"])
    main()
    out = capsys.readouterr().out
    assert "kcai-data-sampling:" in out
    assert "core:" in out
