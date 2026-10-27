"""Frozen benchmark scenarios.

Data only: the S1 batch-size grid and its method parameters, the frozen S2
list of 40 distinct types, and the S3 sigma ladder. Nothing here is derived at
run time — a change to the catalogue or to the pinned upstream must fail an
assertion, never silently change the measured set.
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass

#: S1 batch sizes, one cell each.
S1_BATCH_SIZES: tuple[int, ...] = (1, 5, 10, 20)
S1_BATCH_SIZES = S1_BATCH_SIZES  # backward compat

#: S1's single method, fixed by the benchmark doc.
S1_METHOD: dict[str, Any] = {
    "type": "random_brightness_contrast",
    "seed": 42,
    "brightness_limit": (-0.15, 0.15),
    "contrast_limit": (-0.15, 0.15),
    "ensure_safe_range": True,
    "p": 1.0,
}
B1_METHOD = S1_METHOD  # backward compat

#: S2: 40 distinct registered types, frozen. Families are listed per entry so
#: a catalogue move that changes a family shows up in the manifest diff.
S2_TYPES: tuple[tuple[str, str], ...] = (
    # dihedral (2)
    ("vertical_flip", "dihedral"),
    ("d4", "dihedral"),
    # geometric (8)
    ("rotate", "geometric"),
    ("shift_scale_rotate", "geometric"),
    ("elastic_transform", "geometric"),
    ("random_crop", "geometric"),
    ("pad_if_needed", "geometric"),
    ("perspective", "geometric"),
    ("affine", "geometric"),
    ("safe_rotate", "geometric"),
    # blur (6)
    ("median_blur", "blur"),
    ("motion_blur", "blur"),
    ("unsharp_mask", "blur"),
    ("gaussian_blur", "blur"),
    ("blur", "blur"),
    ("advanced_blur", "blur"),
    # noise (4)
    ("gauss_noise", "noise"),
    ("salt_and_pepper", "noise"),
    ("shot_noise", "noise"),
    ("isonoise", "noise"),
    # color (10)
    ("color_jitter", "color"),
    ("hue_saturation_value", "color"),
    ("random_gamma", "color"),
    ("solarize", "color"),
    ("equalize", "color"),
    ("random_brightness_contrast", "color"),
    ("auto_contrast", "color"),
    ("clahe", "color"),
    ("posterize", "color"),
    ("rgb_shift", "color"),
    # dropout (4)
    ("coarse_dropout", "dropout"),
    ("erasing", "dropout"),
    ("grid_dropout", "dropout"),
    ("pixel_dropout", "dropout"),
    # compression (2)
    ("image_compression", "compression"),
    ("downscale", "compression"),
    # weather (4)
    ("random_fog", "weather"),
    ("random_rain", "weather"),
    ("random_shadow", "weather"),
    ("random_snow", "weather"),
)

#: S2 seed, one per entry (the doc's fixed seed).
S2_SEED = 42

#: S3: sigma = 0.5, 0.7, …, 8.3 — exactly 40 entries.
S3_SIGMAS: tuple[float, ...] = tuple(round(0.5 + 0.2 * i, 1) for i in range(40))

#: S3 seed, fixed for the whole ladder.
S3_SEED = 7

#: The upstream ksize derivation pinned by the benchmark doc.
KSIZE_FACTOR = 3.5


def ksize_for_sigma(sigma: float) -> int:
    """Derive the upstream kernel size for a pinned sigma.

    Args:
        sigma: The pinned sigma.

    Returns:
        The odd kernel size upstream derives: ``int(3.5sigma)·2 + 1``.
    """
    return int(KSIZE_FACTOR * sigma) * 2 + 1


def family_counts() -> dict[str, int]:
    """Count the frozen S2 entries per family.

    Returns:
        Family → number of entries, sorted by family name.
    """
    counts: dict[str, int] = {}
    for _, family in S2_TYPES:
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def b1_cell(batch_size: int) -> dict[str, Any]:
    """Build one S1 cell's job parameters.

    Args:
        batch_size: The cell's B.

    Returns:
        A dict with the batch size and the S1 method config.
    """
    return {
        "bench": "s1",
        "cell": f"s{batch_size}",
        "batch_size": batch_size,
        "method": dict(S1_METHOD),
    }


def b1_cells() -> list[dict[str, Any]]:
    """Build every S1 cell, in batch-size order.

    Returns:
        One dict per S1 cell.
    """
    return [b1_cell(batch_size) for batch_size in S1_BATCH_SIZES]


def s2_entries() -> list[dict[str, Any]]:
    """Build the S2 transformation list.

    Returns:
        One entry per frozen type, with the fixed seed.
    """
    entries: list[dict[str, Any]] = []
    for type_name, _ in S2_TYPES:
        entry: dict[str, Any] = {"name": type_name, "type": type_name, "seed": S2_SEED}
        if type_name == "random_crop":
            entry["height"] = 512
            entry["width"] = 512
        entries.append(entry)
    return entries


def s3_entries() -> list[dict[str, Any]]:
    """Build the S3 transformation ladder.

    Returns:
        One entry per sigma level, in ladder order.
    """
    return [
        {
            "name": f"blur_l{level:02d}",
            "type": "gaussian_blur",
            "seed": S3_SEED,
            "p": 1.0,
            "sigma_limit": [sigma, sigma],
        }
        for level, sigma in enumerate(S3_SIGMAS)
    ]


@dataclass(frozen=True)
class Level:
    """One S3 level, as the ledger and the report see it."""

    level: int
    sigma: float
    ksize: int

    def as_row(self) -> dict[str, Any]:
        """Render the level as a plain row.

        Returns:
            A dict with ``level``, ``sigma`` and ``ksize``.
        """
        return {"level": self.level, "sigma": self.sigma, "ksize": self.ksize}


def s3_levels() -> list[Level]:
    """Build the S3 level table.

    Returns:
        One :class:`Level` per ladder entry.
    """
    return [Level(level=level, sigma=sigma, ksize=ksize_for_sigma(sigma)) for level, sigma in enumerate(S3_SIGMAS)]


def assert_ladder() -> None:
    """Assert the S3 ladder's shape and the derived kernel sizes.

    Raises:
        AssertionError: If the ladder is not 40 entries, not equal-ended, or
            the derived ksize rule does not hold.
    """
    assert len(S3_SIGMAS) == 40, f"S3 ladder must have 40 levels, got {len(S3_SIGMAS)}"
    assert S3_SIGMAS[0] == 0.5, S3_SIGMAS[0]
    assert S3_SIGMAS[-1] == 8.3, S3_SIGMAS[-1]
    for entry in s3_entries():
        low, high = entry["sigma_limit"]
        assert low == high, f"{entry['name']}: sigma_limit must be equal-ended, got {entry['sigma_limit']}"
        assert "blur_limit" not in entry, f"{entry['name']}: blur_limit must stay underived"
    for level in s3_levels():
        assert level.ksize == int(KSIZE_FACTOR * level.sigma) * 2 + 1


def assert_catalogue() -> None:
    """Assert the frozen S2 list against the live registry.

    Imports the registry lazily so importing this module stays cheap and free
    of the albumentations-backed package.

    Raises:
        AssertionError: If a frozen type is missing from the registry, the list
            is not 40 distinct types, or a family's count has drifted.
    """
    from kcai_data_sampling_core.utils.registry import get_transformations_registry

    registry = get_transformations_registry()
    names = [name for name, _ in S2_TYPES]
    assert len(names) == 40, f"S2 must have 40 types, got {len(names)}"
    assert len(set(names)) == 40, "S2 types must be distinct"
    missing = sorted(set(names) - set(registry))
    assert not missing, f"S2 types missing from the registry: {missing}"
    expected = {
        "blur": 6,
        "color": 10,
        "compression": 2,
        "dihedral": 2,
        "dropout": 4,
        "geometric": 8,
        "noise": 4,
        "weather": 4,
    }
    assert family_counts() == expected, f"S2 family coverage drifted: {family_counts()}"


def assert_scenarios() -> None:
    """Run every startup assertion.

    Raises:
        AssertionError: If any frozen scenario has drifted.
    """
    assert len(b1_cells()) == 4, "S1 must have 4 cells"
    assert_ladder()
    assert_catalogue()


# Aliases
S2_ENTRIES = s2_entries
S3_ENTRIES = s3_entries
S3_LEVELS = s3_levels
