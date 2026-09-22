"""CLI utilities: colored console output and a custom logging formatter."""

import logging

from typing_extensions import override


class Bcolors:
    """ANSI color codes for terminal output formatting."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class CustomFormatter(logging.Formatter):
    """Custom log formatter with ANSI color support by log level.

    Attributes:
        MSG_FMT: Default log message format.
    """

    MSG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    LVL_COLOR = {
        logging.DEBUG: Bcolors.OKBLUE,
        logging.INFO: "",
        logging.WARNING: Bcolors.WARNING,
        logging.ERROR: Bcolors.FAIL,
        logging.CRITICAL: Bcolors.FAIL + Bcolors.BOLD,
    }

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with ANSI color based on its log level.

        Args:
            record: The log record to format.

        Returns:
            The formatted message string, with ANSI color codes.
        """
        color = self.LVL_COLOR.get(record.levelno, "")
        log_fmt = color + self.MSG_FMT + Bcolors.ENDC
        return logging.Formatter(log_fmt).format(record)

    @classmethod
    def init_log(cls, level: int | str, format: str = MSG_FMT) -> None:
        """Initialize logging with the custom color formatter.

        Args:
            level: Logging level (int or str).
            format: Log message format string.
        """
        logging.basicConfig(format=format, level=level)
        cls.MSG_FMT = format
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(CustomFormatter())


def main() -> None:
    """Console-script entry point for the umbrella CLI."""
    from kcai_data_sampling.__main__ import execute

    execute()