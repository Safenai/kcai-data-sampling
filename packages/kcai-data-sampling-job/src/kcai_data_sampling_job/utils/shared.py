"""Job helpers shared between the CLI and the orchestrator."""

from kcai_data_sampling_core.models.global_ import ErrorsConfig


def merge_errors(global_errors: ErrorsConfig | None, interface_errors: ErrorsConfig | None) -> ErrorsConfig:
    """Merge global and interface-specific error configs, interface wins.

    Args:
        global_errors: The job-level error policy.
        interface_errors: The interface-level override (may be ``None``).

    Returns:
        A merged ``ErrorsConfig`` with interface fields taking precedence.
    """
    if interface_errors is None:
        return global_errors or ErrorsConfig()
    merged = global_errors or ErrorsConfig()
    if interface_errors.default is not None:
        merged.default = interface_errors.default
    if interface_errors.images is not None:
        merged.images = interface_errors.images
    if interface_errors.tabular is not None:
        merged.tabular = interface_errors.tabular
    if interface_errors.max_failure_rate is not None:
        merged.max_failure_rate = interface_errors.max_failure_rate
    return merged


def parse_memory_string(memory_str: str) -> int:
    """Parse a memory string (e.g. ``"2GB"``, ``"500MB"``) into bytes.

    Args:
        memory_str: Memory string, optionally suffixed with GB/MB/KB/B.

    Returns:
        The memory size in bytes.
    """
    text = memory_str.strip().upper()
    for suffix, factor in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * factor)
    return int(text)