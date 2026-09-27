"""Model resolution in the job CLI.

``cli`` builds the model adapters (``_build_models``) and the transformations
(``_build_transformations``), resolving each model-bearing entry's slot to the
*adapter instance*, never the config's name string. The default env has no
``-lama``, so the tool-role algorithm and the stub adapter are injected for the
duration of one test; the loader and both writers stay the real registry
members. Two end-to-end runs drive the whole ``cli.run`` over the shared
synthetic fixtures: the generative ledger row (``tool_model="stub"``,
``family="generative"``, null ``seed``, ``params`` the region window) and the
adversarial row (``target_model="stub"``, ``family="adversarial"`` — a target
model's role — ``arity=unary``, ``reversible=false``, null ``seed``).

Because ``ToolRoleConfig`` mirrors ``InpaintTransformationConfig`` —
``tool_model`` excluded from the dump — the effect recorded is the
guarantee: resolved ``params`` stay exactly ``{top, left, height, width}``, with
no trace of ``type``, ``seed``, ``storage``, ``columns`` or the model name. The
target role uses the same base machinery: ``TargetRoleTransformation``
(``target_model`` excluded) resolves to the ``StubTarget`` *instance* and its
``params`` are exactly ``{}``, until an algorithm declares its own knobs (the
adversarial one's ``epsilon``). The ``type: python`` branch of ``_build_models``
is exercised here too — a class and a factory constructed with
``weights``+``**params``, an exported instance used as-is, and an empty
``models:`` section building nothing.
"""

import json

import pytest
from pyarrow import parquet as pq

from kcai_data_sampling_core.models.config import JobConfig
from kcai_data_sampling_core.utils.registry import (
    PluginLoadedRegistry,
    get_transformations_registry,
    register_model,
)
from kcai_data_sampling_job import cli
from tests.fixtures.registries import (
    StubTarget,
    StubTool,
    TargetRoleTransformation,
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


def test_build_models_without_a_models_section_is_empty(monkeypatch) -> None:
    """No ``models:`` section builds an empty adapter map.

    ``_build_models`` returns ``{}`` outright — the name resolution in
    ``_build_transformations`` then reports the section as empty rather than
    pretending a model exists.
    """
    validated = _validated(monkeypatch, models=None, adapters={})
    assert cli._build_models(validated) == {}


def test_build_models_python_class_receives_weights_and_params(monkeypatch, tmp_path) -> None:
    """A ``type: python`` class ref is constructed with ``weights`` + ``**params``.

    The exported class is called exactly like a plugin adapter: ``weights`` kept
    only when set, the ref's own ``params`` translated verbatim as its knobs.
    """
    source = tmp_path / "recorder.py"
    source.write_text(
        "class Recorder:\n"
        "    name = 'recorder'\n"
        "    def __init__(self, weights=None, **params):\n"
        "        self.weights = weights\n"
        "        self.params = params\n"
        "    def grad(self, xs):\n"
        "        return xs * 0.0\n",
        encoding="utf-8",
    )
    validated = _validated(
        monkeypatch,
        models={
            "recorder": {
                "type": "python",
                "path": str(source),
                "weights": "ckpt.pt",
                "params": {"margin": 64},
            },
        },
        adapters={},
    )
    built = cli._build_models(validated)
    assert built["recorder"].weights == "ckpt.pt"
    assert built["recorder"].params == {"margin": 64}


def test_build_models_python_class_drops_unset_weights(monkeypatch, tmp_path) -> None:
    """A python class with unset ``weights`` is constructed without that kwarg.

    Mirror of the plugin-route convention: ``weights=None`` is dropped, so a
    constructor that does not accept ``weights`` still works for a bare ref.
    """
    source = tmp_path / "bare.py"
    source.write_text(
        "class Bare:\n"
        "    name = 'bare'\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n"
        "    def grad(self, xs):\n"
        "        return xs * 0.0\n",
        encoding="utf-8",
    )
    validated = _validated(
        monkeypatch,
        models={"bare": {"type": "python", "path": str(source), "params": {"margin": 32}}},
        adapters={},
    )
    built = cli._build_models(validated)
    assert built["bare"].kwargs == {"margin": 32}


def test_build_models_python_factory_is_called_with_weights_and_params(monkeypatch, tmp_path) -> None:
    """A python factory export is called with ``weights`` + ``**params``.

    ``export`` may name a factory callable: construction goes through it with
    the same kwarg conventions as the class route, and the returned adapter is
    the entry's instance.
    """
    source = tmp_path / "factory.py"
    source.write_text(
        "def build(weights=None, **params):\n"
        "    return {'weights': weights, 'params': params}\n",
        encoding="utf-8",
    )
    validated = _validated(
        monkeypatch,
        models={
            "made": {"type": "python", "path": str(source), "export": "build", "params": {"margin": 16}},
        },
        adapters={},
    )
    built = cli._build_models(validated)
    assert built["made"] == {"weights": None, "params": {"margin": 16}}


def test_build_models_python_instance_is_used_as_is(monkeypatch, tmp_path) -> None:
    """A python-exported *instance* is used as-is, never re-constructed.

    The path-derived module cache returns the same module on every load, so the
    exported instance is the very object the source built; the ref's
    ``weights``/``params`` are not applied to it.
    """
    source = tmp_path / "readymade.py"
    source.write_text(
        "class Adapter:\n"
        "    name = 'made'\n"
        "    def __init__(self):\n"
        "        self.kwargs = {'constructed': True}\n"
        "    def grad(self, xs):\n"
        "        return xs * 0.0\n"
        "readymade = Adapter()\n",
        encoding="utf-8",
    )
    validated = _validated(
        monkeypatch,
        models={"made": {"type": "python", "path": str(source), "export": "readymade", "params": {"margin": 8}}},
        adapters={},
    )
    built = cli._build_models(validated)
    assert built["made"].kwargs == {"constructed": True}


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


def test_build_transformations_resolves_the_target_slot(monkeypatch) -> None:
    """The ``target_model`` slot is replaced by the adapter instance.

    The entry says ``target_model: "stub_model"``; the built transformation's
    ``target_model`` is the ``StubTarget`` *instance*, and its resolved
    ``params`` are exactly ``{}`` — the target-role stand-in carries no knobs —
    with no ``target_model``, ``type`` or ``seed`` key surviving resolution.
    """
    validated = _validated(
        monkeypatch,
        models={"stub_model": {"type": "stub"}},
        transformations=[{"type": "test_target", "target_model": "stub_model"}],
        adapters={"stub": StubTarget},
        algorithms={"test_target": TargetRoleTransformation},
    )
    models = cli._build_models(validated)
    instance = cli._build_transformations(validated, get_transformations_registry(), models)[0]
    assert isinstance(instance.target_model, StubTarget)
    assert instance.target_model.name == "stub"
    assert instance.params == {}


def test_build_transformations_refuses_a_missing_target_model_name(monkeypatch) -> None:
    """A target-role transformation that names no model is refused loudly.

    The ``target_model`` key is optional on the config instance, so the refusal
    is the CLI's, and it points at both the missing key and the section to
    choose from.
    """
    validated = _validated(
        monkeypatch,
        models={"stub_model": {"type": "stub"}},
        transformations=[{"type": "test_target"}],
        adapters={"stub": StubTarget},
        algorithms={"test_target": TargetRoleTransformation},
    )
    with pytest.raises(ValueError, match="needs a model \\(target role\\): set 'target_model:'"):
        cli._build_transformations(validated, get_transformations_registry(), cli._build_models(validated))


def test_build_transformations_refuses_an_unknown_target_name_listing_the_section(monkeypatch) -> None:
    """An unknown ``target_model`` name is refused listing every ``models:`` name."""
    validated = _validated(
        monkeypatch,
        models={"stub_model": {"type": "stub"}},
        transformations=[{"type": "test_target", "target_model": "nope"}],
        adapters={"stub": StubTarget},
        algorithms={"test_target": TargetRoleTransformation},
    )
    with pytest.raises(ValueError) as exc:
        cli._build_transformations(validated, get_transformations_registry(), cli._build_models(validated))
    message = str(exc.value)
    assert "unknown target_model model 'nope'" in message
    assert "models: section names: stub_model" in message


def test_build_transformations_refuses_a_target_algorithm_without_models(monkeypatch) -> None:
    """A target-role algorithm with no ``models:`` section is refused (none declared)."""
    validated = _validated(
        monkeypatch,
        models=None,
        transformations=[{"type": "test_target", "target_model": "stub_model"}],
        adapters={},
        algorithms={"test_target": TargetRoleTransformation},
    )
    with pytest.raises(ValueError, match="models: section names: none declared"):
        cli._build_transformations(validated, get_transformations_registry(), {})


def test_cli_run_adversarial_with_registered_stub(registry_snapshot, monkeypatch, raw_bytes_data, tmp_path) -> None:
    """``cli.run`` end to end records the target-role ledger row.

    The full pipeline with the ``register_model``-seeded stub target and the
    injected target-role algorithm emits one row per input carrying
    ``target_model="stub"``, ``family="adversarial"`` (a target's role), a null
    ``seed`` (deterministic algorithm), ``arity=unary`` and an empty
    ``params`` JSON. ``registry_snapshot`` restores the live model registry the
    registration seeded, keeping this test pure.
    """
    register_model("stub", StubTarget)
    set_transformations_registry(monkeypatch, {"test_target": TargetRoleTransformation})
    root = tmp_path / "out"
    config = build_config(
        loaders=[build_loader(parquet_path=raw_bytes_data)],
        output_root=root,
        transformations=[{"type": "test_target", "target_model": "stub_model"}],
        models={"stub_model": {"type": "stub"}},
    )
    assert cli.run(config) == {"synthetic": 8}

    table = pq.read_table(root / "ledger" / "synthetic.parquet")
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    assert len(cols["id"]) == 8
    assert set(cols["target_model"]) == {"stub"}
    assert set(cols["family"]) == {"adversarial"}
    assert set(cols["arity"]) == {"unary"}
    assert set(cols["reversible"]) == {False}
    assert set(cols["seed"]) == {None}
    assert all(json.loads(p) == {} for p in cols["params"])


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