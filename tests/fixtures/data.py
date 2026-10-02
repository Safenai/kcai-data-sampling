"""Synthetic meaningful test data.

Tests never use real frames: every fixture here builds content that is
**non-uniform and visually decodable**, so a failing crop or flip is obvious
to a user looking at a plot or a ledger row. Three deliberate patterns — a
left/right two-tone bar, a checkerboard, and a diagonal gradient — plus a
quadrant pattern for crop visibility, are cycled deterministically over the
rows with a seeded color jitter, so a given ``KCAI_TEST_SEED`` always yields
the same bytes.

The same synthetic frames back the three image-column forms the parquet
loader accepts: **raw bytes** (binary ``img`` + ``height``/``width``
columns), **relative path + prefix**, and **absolute path**. Everything is
regenerated once per seed into ``tests/outputs/data/`` (gitignored) and
reused by every unit/e2e test through the shared fixtures below.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from kcai_data_sampling_images.api.selection import ImageBatch
from kcai_data_sampling_job.dataloaders.api.parquet import ParquetDataLoader, ParquetImageLoaderConfig
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SYNTHETIC_ROWS = 8
SYNTHETIC_SIZE = 32
SELECTION_NAME = "synthetic"
DATASET_NAME = "parquet_synthetic"
ID_COLUMN = "id"
IMG_COLUMN = "img"
PATH_COLUMN = "path"

#: The one inpaint window that fits the 32x32 synthetic frames: a mask
#: inside the frame, non-trivial in both axes, leaving pixels everywhere else
#: untouched for the "unchanged outside the region" assertions.
REGION = {"top": 8, "left": 12, "height": 16, "width": 8}

_PATTERNS = ("bar", "checker", "gradient", "quadrant")


def _jittered(rng: np.random.Generator, base: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """A base RGB color with a per-channel offset drawn from the suite RNG.

    The offset keeps the pattern's identity clear (a small luminance jitter)
    while making the exact bytes depend on the seed.

    Args:
        rng: Seeded generator shared by ``make_frames``.
        base: The pattern's reference RGB color.

    Returns:
        The jittered RGBA color, channels clamped to ``[0, 255]``.
    """
    delta = rng.integers(-10, 11, size=3)
    return tuple(int(np.clip(c + d, 0, 255)) for c, d in zip(base, delta, strict=True)) + (255,)


def _bar(size: int, rng: np.random.Generator) -> np.ndarray:
    """Left/right two-tone frame: a flip visibly swaps the halves."""
    left = _jittered(rng, (220, 50, 50))
    right = _jittered(rng, (50, 50, 220))
    out = np.zeros((size, size, 4), dtype=np.uint8)
    mid = size // 2
    out[:, :mid] = left
    out[:, mid:] = right
    return out


def _checker(size: int, rng: np.random.Generator) -> np.ndarray:
    """4x4 checkerboard: nearest-neighbor resize artifacts stay visible."""
    light = _jittered(rng, (220, 220, 220))
    dark = _jittered(rng, (35, 35, 35))
    out = np.zeros((size, size, 4), dtype=np.uint8)
    block = size // 4
    for i in range(4):
        for j in range(4):
            y0, x0 = i * block, j * block
            out[y0 : y0 + block, x0 : x0 + block] = light if (i + j) % 2 == 0 else dark
    return out


def _gradient(size: int, rng: np.random.Generator) -> np.ndarray:
    """Diagonal gradient: a crop is masked by structure, not uniformity."""
    lo = int(rng.integers(20, 60))
    hi = int(rng.integers(220, 245))
    ramp = np.linspace(lo, hi, size, dtype=np.uint8)
    g = (ramp[:, None] + ramp[None, :]) // 2
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[:, :, 0] = g
    out[:, :, 1] = g // 2
    out[:, :, 2] = (240 - g.astype(np.int16)).astype(np.uint8)
    out[:, :, 3] = 255
    return out


def _quadrant(size: int, rng: np.random.Generator) -> np.ndarray:
    """Four quadrant colors: the crop window maps back to a recognizable spot."""
    q = [
        _jittered(rng, (245, 245, 245)),
        _jittered(rng, (40, 40, 40)),
        _jittered(rng, (40, 120, 40)),
        _jittered(rng, (120, 40, 120)),
    ]
    out = np.zeros((size, size, 4), dtype=np.uint8)
    h = size // 2
    w = size // 2
    out[:h, :w] = q[0]
    out[:h, w:] = q[1]
    out[h:, :w] = q[2]
    out[h:, w:] = q[3]
    return out


_BUILDERS = {
    "bar": _bar,
    "checker": _checker,
    "gradient": _gradient,
    "quadrant": _quadrant,
}


def make_frames(
    seed: int,
    rows: int = SYNTHETIC_ROWS,
    size: int = SYNTHETIC_SIZE,
) -> list[np.ndarray]:
    """One frame per row, pattern cycle deterministic for the given seed.

    Rows cycle through the deliberate patterns in a fixed, inspectable order
    (bar, checker, gradient, quadrant) so row ``i`` is identifiable in plots
    and in the ledger; the seed only drives the per-pattern color jitter.

    Args:
        seed: Suite seed (``KCAI_TEST_SEED``); also keys the on-disk cache.
        rows: Number of frames to build.
        size: Edge length of every frame (all rows share one size).

    Returns:
        ``rows`` RGBA frames, each ``(size, size, 4)`` uint8 in ``[0, 255]``.
    """
    rng = np.random.default_rng(seed)
    return [_BUILDERS[_PATTERNS[i % len(_PATTERNS)]](size, rng) for i in range(rows)]


def write_sidecars(data_dir: Path, frames: list[np.ndarray]) -> None:
    """Write the parquet sidecars and PNG frames for all three column forms.

    ``raw.parquet`` carries the binary ``img`` column plus ``height``/``width``;
    ``relative.parquet`` and ``absolute.parquet`` carry a string ``path`` column
    (relative values resolved against ``prefix``, otherwise absolute). All three
    share the pass-through surface ``id``, ``height``, ``width``, ``source``
    so the forms produce the same ledger rows.

    Args:
        data_dir: Staging directory (``tests/outputs/data``).
        frames: RGBA frames; ``frames[0].shape`` fixes the selection size.
    """
    n = len(frames)
    h, w = frames[0].shape[:2]
    ids = [f"syn_{i:04d}" for i in range(n)]
    frame_dir = data_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    relative: list[str] = []
    absolute: list[str] = []
    for i, frame in enumerate(frames):
        file_name = f"img_{i:02d}.png"
        Image.fromarray(frame, "RGBA").save(frame_dir / file_name)
        relative.append(f"frames/{file_name}")
        absolute.append(str((frame_dir / file_name).resolve()))

    common = {
        ID_COLUMN: ids,
        "height": pa.array([h] * n, type=pa.uint16()),
        "width": pa.array([w] * n, type=pa.uint16()),
        "source": ["synthetic"] * n,
    }
    pq.write_table(pa.table({**common, IMG_COLUMN: [f.tobytes() for f in frames]}), data_dir / "raw.parquet")
    pq.write_table(pa.table({**common, PATH_COLUMN: relative}), data_dir / "relative.parquet")
    pq.write_table(pa.table({**common, PATH_COLUMN: absolute}), data_dir / "absolute.parquet")


@pytest.fixture(scope="session")
def synthetic_data_dir(test_seed: int) -> Path:
    """Session cache of the synthetic images and their sidecar parquet files.

    Regenerates ``tests/outputs/data`` only when the marker for the current
    ``KCAI_TEST_SEED`` is missing (the dqm-ml ``ensure_example_data`` pattern),
    so repeated runs reuse the exact same bytes on disk.

    Returns:
        The staging directory holding ``raw.parquet``, ``relative.parquet``,
        ``absolute.parquet`` and ``frames/*.png``.
    """
    data_dir = Path(__file__).resolve().parents[1] / "outputs" / "data"
    marker = data_dir / f"seed-{test_seed}.ready"
    if not marker.is_file():
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True)
        write_sidecars(data_dir, make_frames(test_seed))
        marker.write_text(json.dumps({"seed": test_seed, "rows": SYNTHETIC_ROWS, "size": SYNTHETIC_SIZE}))
    return data_dir


@pytest.fixture(scope="session")
def raw_bytes_data(synthetic_data_dir: Path) -> Path:
    """Path to the raw-bytes sidecar parquet (binary ``img`` column)."""
    return synthetic_data_dir / "raw.parquet"


@pytest.fixture(scope="session")
def path_data(synthetic_data_dir: Path) -> dict[str, dict[str, str]]:
    """Paths and ``sample_path`` material for the two path column forms.

    ``relative`` pairs the relative-path parquet with the ``prefix`` that
    resolves it (the staging dir); ``absolute`` is the same frames referenced
    by full paths with no prefix needed.

    Returns:
        ``{"relative": {"parquet": ..., "prefix": ...},
        "absolute": {"parquet": ...}}``.
    """
    return {
        "relative": {
            "parquet": str(synthetic_data_dir / "relative.parquet"),
            "prefix": str(synthetic_data_dir),
        },
        "absolute": {
            "parquet": str(synthetic_data_dir / "absolute.parquet"),
        },
    }


@pytest.fixture(scope="session")
def synthetic_selection(raw_bytes_data: Path):
    """A loader-side selection over the synthetic raw-bytes sidecar.

    Built through the real ``parquet`` loader and its registered config schema,
    so loader/CLI wiring tests exercise the exact object a job run reads.

    Returns:
        The selection, named ``synthetic``, one 8-row batch.
    """
    config = ParquetImageLoaderConfig(
        name=SELECTION_NAME,
        path=str(raw_bytes_data),
        sample_path=[{"column": IMG_COLUMN}],
        id_column=ID_COLUMN,
        load_batch_size=SYNTHETIC_ROWS,
    )
    return ParquetDataLoader(name=SELECTION_NAME, config=config).get_selections()[0]


@pytest.fixture(scope="session")
def synthetic_batch(test_seed: int) -> ImageBatch:
    """An ``ImageBatch`` over the synthetic frames, as the loader would emit.

    ``data`` is the stacked RGBA frames; the pass-through ``columns`` mirror
    the loader's batch exactly — ``height``, ``width``, ``source`` — with the
    image column dropped.

    Returns:
        The 8-row ``ImageBatch`` for the current seed.
    """
    frames = make_frames(test_seed)
    n = len(frames)
    h, w = frames[0].shape[:2]
    return ImageBatch(
        name=SELECTION_NAME,
        dataset=DATASET_NAME,
        ids=[f"syn_{i:04d}" for i in range(n)],
        columns=pa.table(
            {
                "height": pa.array([h] * n, type=pa.uint16()),
                "width": pa.array([w] * n, type=pa.uint16()),
                "source": ["synthetic"] * n,
            }
        ),
        data=np.stack(frames),
    )
