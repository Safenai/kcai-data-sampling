"""Shared protocol definitions for data loaders and selections.

The job iterates every selection uniformly: each yields `list[Sample]`
chunks. ``DataSelection`` here is the *source* protocol (disk/table backed);
the transformation engine consumes the in-memory core
``kcai_data_sampling_core.api.selection.DataSelection`` built from each chunk.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataSelection(Protocol):
    """A specific subset of data discovered by a :class:`DataLoader`.

    A selection is a concrete set of samples (a folder, a filtered view of a
    table, a single file) and iterates over in-memory sample batches.

    Attributes:
        name: Unique name of the selection within the job.
        sample_axes: Labels of the axes of each sample's ``x`` array.
        value_range: ``(low, high)`` domain of values of ``x``, or ``None``.
    """

    name: str
    sample_axes: tuple[str, ...]
    value_range: tuple[float, float] | None

    def bootstrap(self, columns_list: list[str] | None) -> None:
        """Perform initial setup for the selection before iteration.

        Args:
            columns_list: Column names to load, or ``None`` for all.
        """

    def get_nb_batches(self) -> int:
        """Return the estimated number of batches of this selection.

        Returns:
            The batch count, used for progress-bar estimation.
        """

    def __iter__(self) -> Any:
        """Iterate over the selection, yielding ``list[Sample]`` chunks."""


@runtime_checkable
class DataLoader(Protocol):
    """Factory protocol: scan a source and discover its data selections.

    A loader is responsible for scanning a source (disk, table) and
    discovering the available :class:`DataSelection` instances.
    """

    def get_selections(self) -> list[DataSelection]:
        """Discover and return the list of available selections.

        Returns:
            A list of initialized selection instances for this loader.
        """