"""Model resolution in the job CLI.

``cli`` builds the model adapters (``_build_models``) and the transformations
(``_build_transformations``), resolving each model-bearing entry's slot to the
*adapter instance*, never the config's name string. The default env has no
``-lama``, so the tool-role algorithm and the stub adapter are injected for the
duration of one test; the loader and both writers stay the real registry
members. The final test drives the whole ``cli.run`` end to end over the shared
synthetic fixtures and asserts the generative ledger row: ``tool_model="stub"``,
``family="generative"`` (tool role → generative), a null ``seed`` (deterministic
algorithm), and ``params`` exactly the region window.

Because ``ToolRoleConfig`` mirrors ``InpaintTransformationConfig`` —
``tool_model`` excluded from the dump — the effect recorded is the
guarantee: resolved ``params`` stay exactly ``{top, left, height, width}``, with
no trace of ``type``, ``seed``, ``storage``, ``columns`` or the model name.
"""

import json

import pytest
from pyarrow import parquet as pq

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_core.utils.registry import PluginLoadedRegistry, get_transformations_registry
from kcai_data_sampling_job import cli
from tests.fixtures.registries import (
    StubTool,
    ToolRoleTransformation,
    set_models_registry,
    set_transformations_registry,
)
from tests.utils.configs import build_config, build_loader


def _validated(
    monkeypatch: pytest.MonkeyPatch,
    *,
    models: dict | None,
    transformations: list[dict] | None = None,
    adapters: dict[str, type],
    algorithms: dict[str, type] | None = None,
) -> JobConfig:
    """Validate a job config against injected registries, without touching disk.

    Args:
        monkeypatch: The test's monkeypatch.
        models: The ``models:`` section contents, or ``None`` to omit it.
        transformations: The transformations list; ``None`` for the builder's
            default procedural trio.
        adapters: Adapter classes keyed by their ``type``, installed as the
            models registry for validation.
        algorithms: Optional transformation registry contents; ``None`` leaves
            the loaded registry in place (the defaults need no injection).

    Returns:
        The validated job configuration.
    """
    set_models_registry(monkeypatch, adapters)
    if algorithms is not None:
        set_transformations_registry(monkeypatch, algorithms)
    return JobConfig.model_validate(
        build_config(
            loaders=[build_loader(parquet_path="missing.parquet")],
            output_path="ledger.parquet",
            samples_dir="samples",
            transformations=transformations,
            models=models,
        )
    )


def test_build_models_drops_unset_weights_and_passes_params(monkeypatch) -> None:
    """Adapters receive ``weights`` only when set, and ``**params`` verbatim.

    ``weights=None`` must be *dropped*: the reference's ``weights`` is unset by
    default, and every adapter's constructor is called without that keyword. The
    plugin's own knobs arrive translated straight from the ``params:`` block.
    """
    captured: list[dict] = []

    class _RecorderTool:
        name = "rec"

        def __init__(self, **kwargs):
            captured.append(kwargs)

        def inpaint(self, xs, masks):
            return xs.copy()

    validated = _validated(
        monkeypatch,
        models={
            "weighted": {"type": "rec", "weights": "checkpoint.pt", "params": {"margin": 64}},
            "bare": {"type": "rec", "params": {"margin": 32}},
        },
        adapters={"rec": _RecorderTool},
    )
    built = cli._build_models(validated)
    assert list(built) == ["weighted", "bare"]
    assert captured == [{"weights": "checkpoint.pt", "margin": 64}, {"margin": 32}]


def test_build_models_refuses_an_unknown_type(monkeypatch) -> None:
    """A ``type`` missing from the models registry is refused loudly.

    Config load already refuses unknown types against the *loaded* registry, so
    this belt-and-braces path is reached when the registry changes between
    validation and construction — the error names every registered adapter.
    """
    validated = _validated(
        monkeypatch,
        models={"m": {"type": "stub"}},
        adapters={"stub": StubTool},
    )
    monkeypatch.setattr(PluginLoadedRegistry, "_models_registry", {})
    with pytest.raises(ValueError, match="model 'm': unknown type 'stub'"):
        cli._build_models(validated)


def test_build_transformations_resolves_slot_and_keeps_params_exact(monkeypatch) -> None:
    """The model-name string is replaced by the adapter instance in the slot.

    The entry says ``tool_model: "stub_model"``; the built transformation's
    ``tool_model`` is the ``StubTool`` *instance*, and its resolved ``params``
    are exactly the region window — no ``tool_model``, ``type``, ``seed``,
    ``storage`` or ``columns`` key survives resolution.
    """
    validated = _validated(
        monkeypatch,
        models={"stub_model": {"type": "stub"}},
        transformations=[
            {"type": "test_tool", "top": 8, "left": 12, "height": 16, "width": 8, "tool_model": "stub_model"}
        ],
        adapters={"stub": StubTool},
        algorithms={"test_tool": ToolRoleTransformation},
    )
    models = cli._build_models(validated)
    instance = cli._build_transformations(validated, get_transformations_registry(), models)[0]
    assert isinstance(instance.tool_model, StubTool)
    assert instance.tool_model.name == "stub"
    assert instance.params == {"top": 8, "left": 12, "height": 16, "width": 8}


def test_build_transformations_refuses_a_missing_model_name(monkeypatch) -> None:
    """A tool-role transformation that names no model is refused loudly.

    The ``tool_model`` key is optional on the config instance (the base
    consumes it before parameter validation), so the refusal is the CLI's, and
    it points at both the missing key and the section to choose from.
    """
    validated = _validated(
        monkeypatch,
        models={"stub_model": {"type": "stub"}},
        transformations=[{"type": "test_tool", "top": 8, "left": 12, "height": 16, "width": 8}],
        adapters={"stub": StubTool},
        algorithms={"test_tool": ToolRoleTransformation},
    )
    with pytest.raises(ValueError, match="needs a model \\(tool role\\): set 'tool_model:'"):
        cli._build_transformations(validated, get_transformations_registry(), cli._build_models(validated))


def test_build_transformations_refuses_an_unknown_name_listing_the_section(monkeypatch) -> None:
    """An unknown model name is refused listing every ``models:`` name.

    The loud message names the alleged model and the complete set of names the
    section offers, so a typo is self-diagnosing.
    """
    validated = _validated(
        monkeypatch,
        models={"stub_model": {"type": "stub"}},
        transformations=[{"type": "test_tool", "top": 8, "left": 12, "height": 16, "width": 8, "tool_model": "nope"}],
        adapters={"stub": StubTool},
        algorithms={"test_tool": ToolRoleTransformation},
    )
    with pytest.raises(ValueError) as exc:
        cli._build_transformations(validated, get_transformations_registry(), cli._build_models(validated))
    message = str(exc.value)
    assert "unknown tool_model model 'nope'" in message
    assert "models: section names: stub_model" in message


def test_build_transformations_refuses_a_role_algorithm_without_models(monkeypatch) -> None:
    """A tool-role algorithm with no ``models:`` section is refused (none declared).

    With no models built, the name resolves to nothing — the error reports the
    section as empty rather than pretending the name exists.
    """
    validated = _validated(
        monkeypatch,
        models=None,
        transformations=[{"type": "test_tool", "top": 8, "left": 12, "height": 16, "width": 8, "tool_model": "stub_model"}],
        adapters={},
        algorithms={"test_tool": ToolRoleTransformation},
    )
    with pytest.raises(ValueError, match="models: section names: none declared"):
        cli._build_transformations(validated, get_transformations_registry(), {})


def test_cli_run_generative_with_injected_stub(monkeypatch, raw_bytes_data, tmp_path) -> None:
    """``cli.run`` end to end records the generative ledger row.

    The full pipeline — the real parquet loader, the stub adapter and the local
    tool-role algorithm injected, the real images and parquet writers — emits
    one row per input carrying ``tool_model="stub"``, ``family="generative"``
    (a tool's role), a null ``seed`` (deterministic algorithm) and the resolved
    region window as its ``params`` JSON.
    """
    set_models_registry(monkeypatch, {"stub": StubTool})
    set_transformations_registry(monkeypatch, {"test_tool": ToolRoleTransformation})
    root = tmp_path / "out"
    config = build_config(
        loaders=[build_loader(parquet_path=raw_bytes_data)],
        output_root=root,
        transformations=[
            {"type": "test_tool", "top": 8, "left": 12, "height": 16, "width": 8, "tool_model": "stub_model"}
        ],
        models={"stub_model": {"type": "stub"}},
    )
    assert cli.run(config) == {"synthetic": 8}

    table = pq.read_table(root / "ledger" / "synthetic.parquet")
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    assert len(cols["id"]) == 8
    assert set(cols["tool_model"]) == {"stub"}
    assert set(cols["family"]) == {"generative"}
    assert set(cols["reversible"]) == {False}
    assert set(cols["seed"]) == {None}
    assert all(json.loads(p) == {"top": 8, "left": 12, "height": 16, "width": 8} for p in cols["params"])