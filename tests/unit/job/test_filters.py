"""Filter-condition building: dispatch on wildcard presence and value shape."""

from kcai_data_sampling_job.dataloaders.filters import build_filter_condition

COLUMN = "label"


def test_list_of_values_with_a_wildcard_goes_to_the_wildcard_function() -> None:
    """Any wildcard member routes the whole list to ``wildcard_fn``."""
    sentinel = object()
    assert (
        build_filter_condition(
            COLUMN,
            ["cat*", "dog"],
            wildcard_fn=lambda col, values: sentinel,
            isin_fn=lambda col, values: object(),
            equal_fn=lambda col, value: object(),
        )
        is sentinel
    )


def test_plain_list_goes_to_the_isin_function() -> None:
    """A wildcard-free list routes to ``isin_fn``."""
    sentinel = object()
    assert (
        build_filter_condition(
            COLUMN,
            ["cat", "dog"],
            wildcard_fn=lambda col, values: object(),
            isin_fn=lambda col, values: sentinel,
            equal_fn=lambda col, value: object(),
        )
        is sentinel
    )


def test_single_value_goes_to_the_equality_function() -> None:
    """A non-list value routes to ``equal_fn``."""
    sentinel = object()
    assert (
        build_filter_condition(
            COLUMN,
            "cat",
            wildcard_fn=lambda col, values: object(),
            isin_fn=lambda col, values: object(),
            equal_fn=lambda col, value: sentinel,
        )
        is sentinel
    )
