"""Where model checkpoints live: ``$KCAI_WEIGHTS_DIR`` or the repository's ``.cache/``.

The default cache directory is the repo-relative ``.cache/`` (discovered by
walking up from the working directory to the nearest ancestor that contains a
``.git`` directory, falling back to the working directory), so a checkout's
checkpoints stay together with its code. ``$KCAI_WEIGHTS_DIR`` overrides it for
shared or pre-fetched caches.
"""

import os
from pathlib import Path
import urllib.request


def weights_path(name: str) -> Path:
    """The cache path of a checkpoint, creating the directory if needed.

    Args:
        name: The checkpoint's file name inside the cache directory.

    Returns:
        The absolute path where the checkpoint is (or will be) cached.
    """
    directory = Path(os.environ.get("KCAI_WEIGHTS_DIR", _default_cache_dir()))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def fetch(name: str, url: str) -> Path:
    """The checkpoint's path, downloaded from ``url`` on first use.

    Args:
        name: The checkpoint's file name inside the cache directory.
        url: The remote location to download from when the file is missing.

    Returns:
        The path the checkpoint was cached at.
    """
    path = weights_path(name)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def _default_cache_dir() -> Path:
    """The repository root's ``.cache/`` (nearest ancestor with a ``.git`` dir).

    Returns:
        ``<repo root>/.cache``, or ``<cwd>/.cache`` when no ancestor has a
        ``.git`` directory.
    """
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists():
            return candidate / ".cache"
    return cwd / ".cache"
