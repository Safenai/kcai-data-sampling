"""Injected plugin registries, the stub tool model, and the cached checkpoint.

The default (torch-free) env has no ``-lama``, so the loaded models registry
is empty and the transformations registry holds only the phase-1 trio. The
mechanism tests in ``unit/core`` and ``unit/job`` pin the validator and
resolution behavior — the
model ``type``/``channels`` validators read the models registry at config-load
time, and ``cli.run`` resolves both registries internally — so those tests
present the registry contents the mechanism expects: a stub adapter (and/or a
role-bearing transformation) installed for the duration of one test via
``monkeypatch``. The opt-in tests run against the real registry and need no
injection.

The stub tool doubles as the resident stand-in for the models registry and for
``Inpaint.apply``'s ``check_output`` path. ``ToolRoleConfig`` /
``ToolRoleTransformation`` are the resident stand-in for a tool-role algorithm
and its pydantic schema: the same role shape as ``Inpaint`` —
``tool`` role, the ``inpaint`` method set, ``tool_model`` excluded from the
dump — without importing ``-lama``. The ``cached_big_lama``
session fixture is opt-in: its lazy import means this module loads in the
default env, but the import and the download only happen when a ``-lama`` test
requests it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import Field
from typing_extensions import Literal

from kcai_data_sampling_core.api.unary import UnaryTransformation
from kcai_data_sampling_core.models.config import TransformationConfig
from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry


class ToolRoleConfig(TransformationConfig):
    """Config schema of the local tool-role test algorithm (mirrors inpaint).

    Subclasses the base ``TransformationConfig`` so the job-level validators
    that read shared keys (``columns``) see them, exactly as an algorithm
    package's own schema does. ``tool_model`` is excluded from the dump (like
    ``InpaintTransformationConfig``), which is what keeps the resolved ``params``
    free of the model-name key.

    Attributes:
        type: Pins the algorithm's registry type.
        tool_model: Optional name of a tool model from the ``models:`` section;
            the base consumes it before parameter validation.
        top, left, height, width: The region-window shape, for the
            resolved-params assertions.
    """

    type: Literal["test_tool"] = "test_tool"
    tool_model: str | None = Field(default=None, exclude=True)
    top: int = 0
    left: int = 0
    height: int = 16
    width: int = 8


class ToolRoleTransformation(UnaryTransformation):
    """Deterministic local stand-in for a model-role algorithm.

    Declares the same role shape as ``Inpaint`` (``tool`` role, the ``inpaint``
    method set) so the config-load and resolution mechanisms are exercised in
    the default env without importing ``-lama``. ``apply`` is the identity:
    this class is only ever *validated* and *resolved* in these tests, never
    meaningfully applied.
    """

    algorithm = "test_tool"
    Config = ToolRoleConfig
    model_role = "tool"
    model_methods = ("inpaint",)
    reversible = False
    stochastic = False

    def apply(self, xs: np.ndarray, rngs: list[np.random.Generator] | None) -> np.ndarray:
        """Map the batch to a copy (identity), honoring the arity contract.

        Args:
            xs: Input batch.
            rngs: Unused; deterministic.

        Returns:
            A copy of ``xs``.
        """
        del rngs
        return xs.copy()


class StubTool:
    """A ``ToolModel``-shaped stand-in: ``name = "stub"``, ``inpaint`` erases the mask.

    ``channels = 3`` mirrors ``LamaTool``'s declared count, so the adapter's
    ``ModelRefConfig`` ``channels``-conflict validator is exercisable in the
    default env. ``inpaint`` zeros only the boolean-mask pixels of the color
    planes and leaves everything else byte-identical — the region-erased
    contract, without LaMa. An empty mask returns a copy untouched.
    """

    channels = 3

    def __init__(self, name: str = "stub") -> None:
        """Build the stub with a row-recordable name.

        Args:
            name: The name recorded on output rows; must be non-empty (the
                ``check_model`` contract).
        """
        if not name:
            raise ValueError("stub tool needs a non-empty name")
        self.name = name

    def inpaint(self, xs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Erase the masked color pixels, leaving the rest (and alpha) intact.

        Args:
            xs: Batch of image arrays shaped ``(B, H, W, C)``.
            masks: One boolean mask per sample ``(B, H, W)``.

        Returns:
            A copy of ``xs`` whose ``True``-masked color pixels are zeroed.
        """
        out = xs.copy()
        for row in range(xs.shape[0]):
            out[row][..., : self.channels][masks[row]] = 0
        return out


def set_models_registry(
    monkeypatch: pytest.MonkeyPatch,
    adapters: dict[str, type] | None = None,
) -> None:
    """Install exact model-registry contents for the duration of one test.

    Args:
        monkeypatch: The test's monkeypatch (restores the attribute after).
        adapters: Adapter classes keyed by their ``type``; defaults to the
            empty registry.
    """
    monkeypatch.setattr(PluginLoadedRegistry, "_models_registry", dict(adapters or {}))


def set_transformations_registry(
    monkeypatch: pytest.MonkeyPatch,
    algorithms: dict[str, type] | None = None,
) -> None:
    """Install exact transformation-registry contents for the duration of one test.

    Args:
        monkeypatch: The test's monkeypatch (restores the attribute after).
        algorithms: Algorithm classes keyed by their ``type``; defaults to the
            empty registry.
    """
    monkeypatch.setattr(PluginLoadedRegistry, "_transformations_registry", dict(algorithms or {}))


@pytest.fixture
def stub_tool() -> StubTool:
    """A fresh stub tool for one test.

    Returns:
        A ``StubTool`` named ``"stub"``.
    """
    return StubTool()


@pytest.fixture(scope="session")
def cached_big_lama() -> Path:
    """The ``big-lama.pt`` checkpoint path, fetched once per session.

    The fixture's sole job is the one allowed suite download: the first-use
    ``fetch`` into ``$KCAI_WEIGHTS_DIR``/``.cache/``. ``-lama`` and torch are
    imported lazily because requests come only from the opt-in modules, which
    the default (torch-free) env skips — so the default suite never downloads
    and never imports torch.

    Returns:
        The cached checkpoint path.
    """
    from kcai_data_sampling_images_lama.api.models.lama import URL
    from kcai_data_sampling_images_lama.api.models.weights import fetch

    return fetch("big-lama.pt", URL)