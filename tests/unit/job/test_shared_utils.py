"""Job shared helpers: error-config merging and memory-string parsing.

``merge_errors`` folds the interface-level error policy over the global one —
each interface field wins when set, and a missing interface falls back to the
global config or the empty defaults. ``parse_memory_string`` turns a suffixed
memory string ("2GB", "500MB", "1KB", "128B") into bytes, case- and
whitespace-insensitive.
"""

from kcai_data_sampling_core.models.global_ import ErrorsConfig, ImageErrorsConfig, TabularErrorsConfig
from kcai_data_sampling_job.utils.shared import merge_errors, parse_memory_string


def test_merge_errors_without_an_interface_returns_the_global_or_default() -> None:
    """A missing interface yields the global policy, else the empty defaults."""
    assert merge_errors(None, None) == ErrorsConfig()
    global_cfg = ErrorsConfig(default="fail_fast", max_failure_rate=0.2)
    assert merge_errors(global_cfg, None) == global_cfg


def test_merge_errors_interface_fields_win_over_the_global() -> None:
    """Every interface field set overrides its global counterpart."""
    global_cfg = ErrorsConfig(default="fail_fast", max_failure_rate=0.2)
    interface = ErrorsConfig(
        default="silent_fail",
        images=ImageErrorsConfig(on_decode_failure="fail_fast"),
        tabular=TabularErrorsConfig(on_missing_column="silent_fail"),
        max_failure_rate=0.1,
    )
    merged = merge_errors(global_cfg, interface)
    assert merged is global_cfg
    assert merged.default == "silent_fail"
    assert merged.images == ImageErrorsConfig(on_decode_failure="fail_fast")
    assert merged.tabular == TabularErrorsConfig(on_missing_column="silent_fail")
    assert merged.max_failure_rate == 0.1


def test_parse_memory_string_handles_every_suffix() -> None:
    """GB/MB/KB/B suffixes (case-insensitive) and a bare number all parse."""
    assert parse_memory_string("2GB") == 2 * 1024**3
    assert parse_memory_string(" 500Mb ") == 500 * 1024**2
    assert parse_memory_string("1KB") == 1024
    assert parse_memory_string("128B") == 128
    assert parse_memory_string("0.5GB") == int(0.5 * 1024**3)
    assert parse_memory_string("1024") == 1024
