"""Generic protocols for data loaders and selections.

A loader scans a source (disk, table) and discovers its data selections; a
selection iterates over in-memory :class:`Batch` chunks (batch = row), ready
for the transformation engine. The contracts are datatype-agnostic — nothing
here assumes images; datatype packages fix the sample's axes and value range
on their batch subclass.
"""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from kcai_data_sampling_core.api.selection import Batch


@runtime_checkable
class DataSelection(Protocol):
    """A specific subset of data discovered by a :class:`DataLoader`.

    A selection is a concrete set of samples (a filtered view of a table) and
    iterates over in-memory batches.

    Attributes:
        name: Unique name of the selection within the job.
        dataset: Name of the reader (loader) that assembled the selection.
        sample_axes: Labels of the axes of each row's sample array, or ``None``.
        value_range: ``(low, high)`` domain of values of each row, or ``None``.
    """

    name: str
    dataset: str
    sample_axes: tuple[str, ...] | None
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

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over the selection, yielding ``Batch`` chunks."""
        ...


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
