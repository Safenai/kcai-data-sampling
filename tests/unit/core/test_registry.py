"""Registry resolution and interface-split guards.

These tests pin the plugin wiring: each of the four entry-point groups resolves
the right objects from the right packages, and the split itself survives — core
exposes the generic interface only, ``ImageBatch`` is the image specialization,
and ``-job`` (the loader/writer owner) never imports ``-images``.
"""

import inspect
import re
from pathlib import Path

import pytest

import kcai_data_sampling_core.api as core_api
import kcai_data_sampling_job
from kcai_data_sampling_core.utils.registry import (
    PluginLoadedRegistry,
    get_transformations_registry,
)
from kcai_data_sampling_images.api.selection import ImageBatch
from kcai_data_sampling_job.dataloaders.api.parquet import ParquetDataLoader

_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+kcai_data_sampling_images", re.MULTILINE)


def test_transformations_registry_resolves_to_images_package() -> None:
    """The transformations group resolves the image algorithms from image packages.

    Every registered algorithm class carries its ``api.transformations`` module
    in a ``-images*`` package (core or its plugins), so the generic core never
    has to know the specific image packages at import time. Env-agnostic: the
    phase-1 trio is a required subset in any env, and the opt-in ``inpaint``
    entry (``-images-lama``) is simply another member when installed.
    """
    registry = get_transformations_registry()
    assert {"horizontal_flip", "crop_resize"} <= set(registry)
    assert all(cls.__module__.startswith("kcai_data_sampling_images") for cls in registry.values())


def test_dataloaders_registry_resolves_parquet_from_job() -> None:
    """The dataloaders group resolves the parquet loader from -job."""
    registry = PluginLoadedRegistry.get_dataloaders_registry()
    assert registry == {"parquet": ParquetDataLoader}
    assert registry["parquet"].__module__.startswith("kcai_data_sampling_job")


def test_outputwriter_registry_resolves_writers_from_job() -> None:
    """The outputwriter group resolves the images/parquet writers from -job.

    The two writers (payload + ledger) live next to the loader in ``-job``;
    core only ships the generic ``OutputWriter`` protocol.
    """
    registry = PluginLoadedRegistry.get_outputwriter_registry()
    assert set(registry) == {"images", "parquet"}
    assert all(cls.__module__.startswith("kcai_data_sampling_job") for cls in registry.values())


def test_models_registry_resolves_only_registered_model_plugins() -> None:
    """The models group holds exactly the installed model adapters, from plugin packages.

    The group must load cleanly with zero adapters (the default torch-free env
    has no model package installed) and never smuggle in a core-internal
    adapter: every value is a class whose module lives in a ``kcai_data_sampling_*``
    package. The exact membership of the opt-in env is pinned by
    ``tests/unit/lama/test_registry_extras.py`` (``{"lama_inpaint"}``).
    """
    registry = PluginLoadedRegistry.get_models_registry()
    assert all(cls.__module__.startswith("kcai_data_sampling_") for cls in registry.values())


def test_core_api_exports_no_imagebatch() -> None:
    """The split guard: core's generic API surface never leaks ImageBatch.

    ``-images`` is a consumer of core, not the other way around; a test or a
    future n-ary algorithm reaches the image batch through
    ``kcai_data_sampling_images.api.selection`` only.
    """
    assert not hasattr(core_api, "ImageBatch")
    assert "ImageBatch" not in core_api.__all__
    assert set(core_api.__all__) == {
        "Batch",
        "DataLoader",
        "DataSelection",
        "FAMILY_BY_ROLE",
        "Output",
        "OutputWriter",
        "TargetModel",
        "ToolModel",
        "Transformation",
        "UnaryTransformation",
    }


def test_imagebatch_specializes_batch_with_fixed_image_space() -> None:
    """ImageBatch fixes the generic Batch's optional image space.

    The split pins the image layout: ``sample_axes`` names ``(H, W, C)`` in
    that order and ``value_range`` is the uint8 ``[0, 255]`` domain — the
    contracts the fixtures and the algorithm tests rely on.
    """
    from kcai_data_sampling_core.api.selection import Batch

    assert issubclass(ImageBatch, Batch)
    assert ImageBatch.sample_axes == ("height", "width", "channel")
    assert ImageBatch.value_range == (0.0, 255.0)


def test_job_package_source_never_imports_images() -> None:
    """Split guard: -job's loader/writer plugins import core only, never -images.

    The static check scans the installed ``-job`` sources for an import of
    ``kcai_data_sampling_images``; a hit would mean the generic engine gained a
    hidden image dependency (the loader and writers must stay datatype-agnostic
    and swapable). Import *uses* (``ImageBatch`` from ``-images``) belong in
    ``-images``' own tests and fixtures.
    """
    package_dir = Path(inspect.getfile(kcai_data_sampling_job)).resolve().parent
    hits = [str(p) for p in package_dir.rglob("*.py") if _IMPORT_RE.search(p.read_text(encoding="utf-8"))]
    assert hits == [], f"-job sources import kcai_data_sampling_images: {hits}"


def test_transformations_are_registry_reachable_by_type_string() -> None:
    """Every transformation is nameable by its config ``type`` string.

    The CLI resolves entries ``{"type": ...}`` against this registry; the two
    image algorithms respond to their documented type literals.
    """
    assert get_transformations_registry()["horizontal_flip"].algorithm == "horizontal_flip"
    assert get_transformations_registry()["crop_resize"].algorithm == "crop_resize"