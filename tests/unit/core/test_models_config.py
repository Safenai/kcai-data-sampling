"""Model references and the ``models:`` job section.

The default env has no model package installed, so the models registry is
empty and the ``type``/``channels`` validators — which read it at config-load
time — get their contents injected (``set_models_registry``). The scope here
is the *load* contract: an unknown ``type`` is refused loudly, a ``channels``
conflict with the adapter's declared count is refused, the flat and nested
``models:`` shapes validate to the same mapping, and a ``JobConfig`` carrying
a ``models:`` section plus a role-requiring transformation loads whether or
not the transformation names its model — the missing *name* is a job-start
error, resolved by the CLI, not a load error.
"""

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_core.models.models import ModelRefConfig, ModelsConfig
import pydantic
import pytest

from tests.fixtures.data import IMG_COLUMN
from tests.fixtures.registries import (
    StubTool,
    ToolRoleTransformation,
    set_models_registry,
    set_transformations_registry,
)
from tests.utils.configs import build_config, build_loader


class _NeutralTool:
    """A ``ToolModel``-shaped adapter declaring no channel count.

    The ``channels``-conflict validator only fires when an adapter declares a
    pinned count; this neutral stand-in is how the "override accepted" paths
    of ``channels`` (which use the adapter's declared value as default) are
    exercised without collision.
    """

    name = "neutral"

    def inpaint(self, xs, masks):
        return xs.copy()


def test_model_ref_unknown_type_is_refused_loudly(monkeypatch) -> None:
    """A model ``type`` outside the loaded registry fails at load, listing knowns.

    The ``_known`` validator reads the *installed* registry, so the injected
    adapters are what it lists — the loud message names every registered
    adapter so a typo is self-diagnosing.
    """
    set_models_registry(monkeypatch, {"lama_inpaint": StubTool, "other_tool": StubTool})
    with pytest.raises(pydantic.ValidationError) as exc:
        ModelRefConfig(type="no_such_tool")
    message = str(exc.value)
    assert "unknown model type 'no_such_tool'" in message
    assert "lama_inpaint" in message
    assert "other_tool" in message


def test_model_ref_channels_is_limited_to_rgb_or_rgba(monkeypatch) -> None:
    """``channels`` accepts only 3 or 4 (or unset); anything else is refused.

    The adapter's declared count is the contract defaults are read from, so
    matching overrides are the explicit form, and out-of-domain values fail at
    the ``Literal`` bound — before any conflict logic runs. The neutral adapter
    (no declared count) accepts either 3 or 4.
    """
    set_models_registry(monkeypatch, {"neutral": _NeutralTool})
    assert ModelRefConfig(type="neutral").channels is None
    assert ModelRefConfig(type="neutral", channels=3).channels == 3
    assert ModelRefConfig(type="neutral", channels=4).channels == 4
    with pytest.raises(pydantic.ValidationError, match="Input should be 3 or 4"):
        ModelRefConfig(type="neutral", channels=5)


def test_model_ref_refuses_a_channels_conflict_with_the_adapter(monkeypatch) -> None:
    """An override disagreeing with the adapter's declared count is refused.

    ``StubTool`` declares ``channels = 3``; asking for the RGBA plane (4) for
    an RGB-only adapter is a config error that names both values so the intent
    is clear.
    """
    set_models_registry(monkeypatch, {"stub": StubTool})
    assert ModelRefConfig(type="stub", channels=3).channels == 3
    with pytest.raises(pydantic.ValidationError) as exc:
        ModelRefConfig(type="stub", channels=4)
    message = str(exc.value)
    assert "declares channels=3" in message
    assert "channels=4" in message


def test_models_config_flat_and_nested_sections_validate_to_the_same_mapping(monkeypatch) -> None:
    """``models: {name: ref}`` and ``models: {models: {name: ref}}`` agree.

    The flat YAML section is the documented spelling; the doubly-nested form
    keeps working as the raw schema shape. Both must resolve to the identical
    name→reference mapping.
    """
    set_models_registry(monkeypatch, {"stub": StubTool})
    flat = ModelsConfig.model_validate({"lama": {"type": "stub", "weights": "big-lama.pt"}})
    nested = ModelsConfig.model_validate({"models": {"lama": {"type": "stub", "weights": "big-lama.pt"}}})
    assert flat.models == nested.models
    assert list(flat.models) == ["lama"]
    assert flat.models["lama"].type == "stub"
    assert flat.models["lama"].weights == "big-lama.pt"


def test_job_config_loads_with_a_models_section_and_a_role_transformation(monkeypatch, tmp_path) -> None:
    """A role-requiring transformation validates at config load with ``models:``.

    A ``tool``-role transformation whose config *names* its model loads cleanly:
    the reference resolves against the injected models registry, and the entry
    is validated against the algorithm's own schema.
    """
    set_models_registry(monkeypatch, {"stub": StubTool})
    set_transformations_registry(monkeypatch, {"test_tool": ToolRoleTransformation})
    config = build_config(
        loaders=[build_loader(parquet_path="missing.parquet")],
        output_path=str(tmp_path / "ledger.parquet"),
        samples_dir=str(tmp_path / "s"),
        transformations=[{"type": "test_tool", "tool_model": "stub_model"}],
        models={"stub_model": {"type": "stub"}},
    )
    validated = JobConfig.model_validate(config)
    assert len(validated.operations.transformations) == 1
    entry = validated.operations.transformations[0]
    assert entry.type == "test_tool"
    assert entry.tool_model == "stub_model"
    assert set(validated.models.models) == {"stub_model"}
    ref = validated.models.models["stub_model"]
    assert (ref.type, ref.weights, ref.params, ref.channels) == ("stub", None, {}, None)


def test_job_config_missing_model_name_is_a_start_error_not_a_load_error(monkeypatch, tmp_path) -> None:
    """A role transformation without a model name *loads* — the CLI refuses it.

    ``tool_model`` is optional on the config instance (the base consumes it
    before parameter validation), so validation cannot know which ``models:``
    name to require. The loud refusal at job start is the CLI's job.
    """
    set_models_registry(monkeypatch, {"stub": StubTool})
    set_transformations_registry(monkeypatch, {"test_tool": ToolRoleTransformation})
    config = build_config(
        loaders=[build_loader(parquet_path="missing.parquet")],
        output_path=str(tmp_path / "ledger.parquet"),
        samples_dir=str(tmp_path / "s"),
        transformations=[{"type": "test_tool"}],
        models={"stub_model": {"type": "stub"}},
    )
    validated = JobConfig.model_validate(config)
    assert validated.operations.transformations[0].tool_model is None


def test_job_config_columns_input_must_name_exactly_one_column(monkeypatch, tmp_path) -> None:
    """``columns.input`` with two names is refused at load, naming what it got."""
    set_transformations_registry(monkeypatch, {"test_tool": ToolRoleTransformation})
    config = build_config(
        loaders=[build_loader(parquet_path="missing.parquet")],
        output_root=str(tmp_path),
        transformations=[{"type": "test_tool", "name": "tool", "columns": {"input": [IMG_COLUMN, "other"]}}],
    )
    with pytest.raises(pydantic.ValidationError) as exc:
        JobConfig.model_validate(config)
    assert "columns.input must name exactly one column" in str(exc.value)


def test_job_config_columns_input_must_name_a_sample_path_column(monkeypatch, tmp_path) -> None:
    """``columns.input`` naming no loader's ``sample_path.column`` is refused."""
    set_transformations_registry(monkeypatch, {"test_tool": ToolRoleTransformation})
    config = build_config(
        loaders=[build_loader(parquet_path="missing.parquet")],
        output_root=str(tmp_path),
        transformations=[{"type": "test_tool", "name": "tool", "columns": {"input": ["other"]}}],
    )
    with pytest.raises(pydantic.ValidationError) as exc:
        JobConfig.model_validate(config)
    message = str(exc.value)
    assert "columns.input 'other' is not any loader's sample_path.column" in message
    assert "loaders: ['img']" in message


def test_job_config_columns_input_naming_the_sample_column_loads(monkeypatch, tmp_path) -> None:
    """``columns.input: [img]`` matching the loader's column validates cleanly."""
    set_transformations_registry(monkeypatch, {"test_tool": ToolRoleTransformation})
    config = build_config(
        loaders=[build_loader(parquet_path="missing.parquet")],
        output_root=str(tmp_path),
        transformations=[{"type": "test_tool", "name": "tool", "columns": {"input": [IMG_COLUMN]}}],
    )
    validated = JobConfig.model_validate(config)
    assert validated.operations.transformations[0].columns.input == [IMG_COLUMN]
