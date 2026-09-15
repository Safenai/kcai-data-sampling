"""Where checkpoints live: ``$KCAI_WEIGHTS_DIR`` or ``~/.cache/kcai-data-sampling/weights``."""

import os
import urllib.request
from pathlib import Path


def weights_path(name: str) -> Path:
    directory = Path(os.environ.get("KCAI_WEIGHTS_DIR", Path.home() / ".cache" / "kcai-data-sampling" / "weights"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def fetch(name: str, url: str) -> Path:
    """The checkpoint's path, downloaded from ``url`` on first use."""
    path = weights_path(name)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path
