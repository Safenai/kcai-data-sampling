"""Frozen benchmark scenarios.

Data only: the B1 batch-size grid and its method parameters, the frozen B2-a
list of 40 distinct types, and the B2-b sigma ladder. Nothing here is derived at
run time — a change to the catalogue or to the pinned upstream must fail an
assertion, never silently change the measured set.
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass

#: B1 batch sizes, one cell each.
B1_BATCH_SIZES: tuple[int, ...] = (1, 5, 10, 20)

#: B1's single method, fixed by the benchmark doc.
B1_METHOD: dict[str, Any] = {
    "type": "random_brightness_contrast",
    "seed": 42,
    "brightness_limit": [-0.2, 0.2],
    "contrast_limit": [-0.2, 0.2],
    "ensure_safe_range": False,
    "p": 1.0,
}

#: B2-a: 40 distinct registered types, frozen. Families are listed per entry so
#: a catalogue move that changes a family shows up in the manifest diff.
B2A_TYPES: tuple[tuple[str, str], ...] = (
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

#: B2-a seed, one per entry (the doc's fixed seed).
B2A_SEED = 42

#: B2-b: sigma = 0.5, 0.7, …, 8.3 — exactly 40 entries.
B2B_SIGMAS: tuple[float, ...] = tuple(round(0.5 + 0.2 * i, 1) for i in range(40))

#: B2-b seed, fixed for the whole ladder.
B2B_SEED = 7

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
    """Count the frozen B2-a entries per family.

    Returns:
        Family → number of entries, sorted by family name.
    """
    counts: dict[str, int] = {}
    for _, family in B2A_TYPES:
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def b1_cell(batch_size: int) -> dict[str, Any]:
    """Build one B1 cell's job parameters.

    Args:
        batch_size: The cell's B.

    Returns:
        A dict with the batch size and the B1 method config.
    """
    return {
        "bench": "b1",
        "cell": f"b{batch_size}",
        "batch_size": batch_size,
        "method": dict(B1_METHOD),
    }


def b1_cells() -> list[dict[str, Any]]:
    """Build every B1 cell, in batch-size order.

    Returns:
        One dict per B1 cell.
    """
    return [b1_cell(batch_size) for batch_size in B1_BATCH_SIZES]


def b2a_entries() -> list[dict[str, Any]]:
    """Build the B2-a transformation list.

    Returns:
        One entry per frozen type, with the fixed seed.
    """
    entries: list[dict[str, Any]] = []
    for type_name, _ in B2A_TYPES:
        entry: dict[str, Any] = {"name": type_name, "type": type_name, "seed": B2A_SEED}
        if type_name == "random_crop":
            entry["height"] = 512
            entry["width"] = 512
        entries.append(entry)
    return entries


def b2b_entries() -> list[dict[str, Any]]:
    """Build the B2-b transformation ladder.

    Returns:
        One entry per sigma level, in ladder order.
    """
    return [
        {
            "name": f"blur_l{level:02d}",
            "type": "gaussian_blur",
            "seed": B2B_SEED,
            "p": 1.0,
            "sigma_limit": [sigma, sigma],
        }
        for level, sigma in enumerate(B2B_SIGMAS)
    ]


@dataclass(frozen=True)
class Level:
    """One B2-b level, as the ledger and the report see it."""

    level: int
    sigma: float
    ksize: int

    def as_row(self) -> dict[str, Any]:
        """Render the level as a plain row.

        Returns:
            A dict with ``level``, ``sigma`` and ``ksize``.
        """
        return {"level": self.level, "sigma": self.sigma, "ksize": self.ksize}


def b2b_levels() -> list[Level]:
    """Build the B2-b level table.

    Returns:
        One :class:`Level` per ladder entry.
    """
    return [Level(level=level, sigma=sigma, ksize=ksize_for_sigma(sigma)) for level, sigma in enumerate(B2B_SIGMAS)]


def assert_ladder() -> None:
    """Assert the B2-b ladder's shape and the derived kernel sizes.

    Raises:
        AssertionError: If the ladder is not 40 entries, not equal-ended, or
            the derived ksize rule does not hold.
    """
    assert len(B2B_SIGMAS) == 40, f"B2-b ladder must have 40 levels, got {len(B2B_SIGMAS)}"
    assert B2B_SIGMAS[0] == 0.5, B2B_SIGMAS[0]
    assert B2B_SIGMAS[-1] == 8.3, B2B_SIGMAS[-1]
    for entry in b2b_entries():
        low, high = entry["sigma_limit"]
        assert low == high, f"{entry['name']}: sigma_limit must be equal-ended, got {entry['sigma_limit']}"
        assert "blur_limit" not in entry, f"{entry['name']}: blur_limit must stay underived"
    for level in b2b_levels():
        assert level.ksize == int(KSIZE_FACTOR * level.sigma) * 2 + 1


def assert_catalogue() -> None:
    """Assert the frozen B2-a list against the live registry.

    Imports the registry lazily so importing this module stays cheap and free
    of the albumentations-backed package.

    Raises:
        AssertionError: If a frozen type is missing from the registry, the list
            is not 40 distinct types, or a family's count has drifted.
    """
    from kcai_data_sampling_core.utils.registry import get_transformations_registry

    registry = get_transformations_registry()
    names = [name for name, _ in B2A_TYPES]
    assert len(names) == 40, f"B2-a must have 40 types, got {len(names)}"
    assert len(set(names)) == 40, "B2-a types must be distinct"
    missing = sorted(set(names) - set(registry))
    assert not missing, f"B2-a types missing from the registry: {missing}"
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
    assert family_counts() == expected, f"B2-a family coverage drifted: {family_counts()}"


def assert_scenarios() -> None:
    """Run every startup assertion.

    Raises:
        AssertionError: If any frozen scenario has drifted.
    """
    assert len(b1_cells()) == 4, "B1 must have 4 cells"
    assert_ladder()
    assert_catalogue()
