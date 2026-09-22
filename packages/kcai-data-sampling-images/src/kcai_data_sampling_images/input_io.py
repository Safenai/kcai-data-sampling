"""The ``image_dir`` dataloader: one selection = the image files of a folder.

Decode-on-demand streaming: each chunk decodes its files **once** into
``(H, W, 4)`` uint8 RGBA views, builds the sample rows, and drops the decoded
bytes when the chunk is done. Idempotency: ``source``
carries the absolute file path, so a recipe re-run on the same folder hashes
to the same payloads.
"""

import glob
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from kcai_data_sampling_core.api.selection import DataSelection, Sample
from kcai_data_sampling_core.models.dataloaders import DataLoaderConfig
from kcai_data_sampling_images.utils.images import SUPPORTED_EXTENSIONS, decode_image

DEFAULT_ERRORS: dict[str, str] = {
    "on_decode_failure": "silent_fail",
    "on_unsupported_format": "fail_fast",
}


def _discover_files(path: str, on_unsupported_format: str = "fail_fast") -> list[str]:
    """Return the sorted image file paths for a directory or glob pattern.

    Args:
        path: A directory or a glob pattern.
        on_unsupported_format: Error policy for files with an unsupported
            extension: ``fail_fast`` raises, ``silent_fail`` skips them.

    Returns:
        Sorted list of file paths with a supported image extension.

    Raises:
        RuntimeError: If a file has an unsupported extension and the policy is
            ``fail_fast``.
    """
    if any(ch in path for ch in "*?["):
        raw = glob.glob(path)
    else:
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(f"image_dir: no such directory: {path}")
        raw = [str(p) for p in sorted(directory.iterdir()) if p.is_file()]
    discovered: list[str] = []
    for p in sorted(raw):
        if Path(p).suffix in SUPPORTED_EXTENSIONS:
            discovered.append(p)
        elif on_unsupported_format == "fail_fast":
            raise RuntimeError(f"image_dir: unsupported image format: {p}")
    return discovered


class ImageDirDataLoader:
    """Loads the image files of a folder as a single selection.

    The plugin for the ``image_dir`` registered type
    (``kcai_data_sampling.dataloaders`` entry point).

    Attributes:
        name: Unique dataloader name (the selection name).
        errors: The active image error policy.
    """

    def __init__(
        self,
        name: str,
        config: DataLoaderConfig,
        errors: dict[str, str] | None = None,
        threads: int = 4,
    ):
        """Build the loader for a folder of images.

        Args:
            name: Dataloader name; names the single selection.
            config: The dataloader configuration; ``path`` (directory or
                glob), ``load_batch_size``, ``decode`` and ``id_column`` are
                honored. ``decode`` must be ``"img_bytes"`` this phase —
                anything else is refused.
            errors: Image error policy override; defaults to
                :data:`DEFAULT_ERRORS`.
            threads: Ignored; retained for a uniform loader constructor
                signature.

        Raises:
            ValueError: If ``decode`` is not ``"img_bytes"``.
        """
        if config.decode and config.decode != "img_bytes":
            raise ValueError(
                f"image_dir: decode mode {config.decode!r} is not implemented this phase."
            )
        self.name = name
        self.path = config.path
        self.load_batch_size = config.load_batch_size
        self.id_column = config.id_column
        self.errors = {**DEFAULT_ERRORS, **(errors or {})}
        del threads

    def get_selections(self) -> list["ImageDirDataSelection"]:
        """Discover the folder's image files.

        Returns:
            A single selection whose samples stream from the files.
        """
        return [
            ImageDirDataSelection(
                self,
                _discover_files(self.path, self.errors["on_unsupported_format"]),
            )
        ]


class ImageDirDataSelection:
    """The streaming selection behind an ``image_dir`` loader.

    Iteration yields one `list[Sample]` per ``load_batch_size`` chunk; each
    Sample's ``x`` is a zero-copy row view of the chunk's decoded RGBA bytes.
    """

    sample_axes: tuple[str, ...] = ("height", "width", "channel")
    value_range: tuple[float, float] = (0.0, 255.0)

    def __init__(self, loader: ImageDirDataLoader, files: list[str]):
        """Build the selection.

        Args:
            loader: The owning dataloader (its name, batch size and error
                policy are used).
            files: The sorted image file paths.
        """
        self.name = loader.name
        self.dataset = loader.name
        self.loader = loader
        self.files = files
        self._failures = 0

    @property
    def nb_samples(self) -> int:
        """Number of files streamed by this selection."""
        return len(self.files)

    def bootstrap(self, columns: list[str] | None = None) -> None:
        """Prepare the selection for iteration; a no-op for image files.

        Args:
            columns: Ignored; image selections carry their own columns.
        """
        del columns

    def get_nb_batches(self) -> int:
        """Compute the number of batches yielded by :meth:`__iter__`.

        Returns:
            ``ceil(nb_samples / load_batch_size)``.
        """
        size = self.loader.load_batch_size
        return (len(self.files) + size - 1) // size

    @property
    def failures(self) -> int:
        """Number of rows skipped because a file could not be decoded."""
        return self._failures

    def __iter__(self) -> Iterator[list[Sample]]:
        """Yield one chunk of samples per batch.

        Chunks honor ``load_batch_size``; on each chunk the files are decoded
        once into RGBA bytes and dropped when the chunk is done.
        """
        size = self.loader.load_batch_size
        for start in range(0, len(self.files), size):
            yield self._decode_chunk(self.files[start : start + size])

    def _decode_chunk(self, files: list[str]) -> list[Sample]:
        """Decode one chunk of files into sample rows.

        Args:
            files: The file paths of the chunk.

        Returns:
            One ``Sample`` per decodable file, in file order.

        Raises:
            RuntimeError: If the error policy is ``fail_fast`` and a decode
                fails.
        """
        rows: list[Sample] = []
        for path in files:
            try:
                height, width, rgba_bytes = decode_image(path)
            except ValueError:
                self._failures += 1
                if self.errors["on_decode_failure"] == "fail_fast":
                    raise
                continue
            x = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(height, width, 4)
            rows.append(
                Sample(
                    id=Path(path).stem,
                    x=x,
                    source={"path": str(Path(path).resolve())},
                )
            )
        return rows