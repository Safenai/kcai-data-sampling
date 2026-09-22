"""Main CLI entry point: version, list, process.

Port of dqm-ml's umbrella ``__main__.py``: a subcommand dispatcher where the
``process`` command only appears when ``kcai-data-sampling-job`` is installed
(wiring).
"""

import argparse
from collections.abc import Iterable
import logging
from typing import Any

from typing_extensions import override

from kcai_data_sampling.cli_tools import CustomFormatter
from kcai_data_sampling.dependency import get_available_command

logger = logging.getLogger(__name__)


class _HelpAction(argparse._HelpAction):
    """Custom help action supporting per-command help."""

    @override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Iterable[Any] | None,
        option_string: str | None = None,
    ) -> None:
        """Handle help for a specific command or the global help.

        Args:
            parser: The argument parser instance.
            namespace: Parsed namespace containing the command.
            values: Optional values passed to the help action.
            option_string: The option string that triggered this action.

        Raises:
            ValueError: If the named command is unknown.
        """
        if namespace.command:
            command_list = get_available_command()
            if namespace.command in command_list and command_list[namespace.command] is not None:
                command_list[namespace.command](["-h"])
            else:
                raise ValueError(f"Unknown command {namespace.command}")
        else:
            parser.print_help()
            parser.exit()


def parse_args(arg_list: list[str] | None, command_list: Iterable[str]) -> Any:
    """Parse the umbrella CLI arguments.

    Args:
        arg_list: The raw argument list (or ``None`` for ``sys.argv``).
        command_list: Iterable of available command names.

    Returns:
        A tuple of ``(parsed_args, remaining_args)``.
    """
    parser = argparse.ArgumentParser(
        prog="kcai-data-sampling",
        description="kcai data-sampling job client",
        epilog="for more information see README",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action=_HelpAction, help="print the help for help")
    parser.add_argument("command", choices=list(command_list), help="available command for kcai data-sampling")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="errors only")
    cli_args, remaining = parser.parse_known_args(arg_list)
    return cli_args, remaining


def execute(arg_list: list[str] | None = None) -> None:
    """Dispatch to the requested CLI command.

    Args:
        arg_list: The raw argument list (or ``None`` for ``sys.argv``).

    Raises:
        ValueError: If the command is unknown.
    """
    command_list = get_available_command()
    args, remaining = parse_args(arg_list, command_list)

    if args.verbose:
        CustomFormatter.init_log(format="%(name)s - %(message)s (%(filename)s:%(lineno)d)", level=logging.DEBUG)
    elif args.quiet:
        CustomFormatter.init_log(format="%(message)s", level=logging.ERROR)
    else:
        CustomFormatter.init_log(format="%(message)s", level=logging.INFO)

    logger.debug("Executing kcai-data-sampling with %s", arg_list)

    if args.command in command_list and command_list[args.command] is not None:
        command_list[args.command](remaining)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    execute()