"""Extras smoke: what each umbrella extra does, and does not, pull from the index.

Runs inside one ``umbrella+<extra>`` venv, installed from an index as
``kcai-data-sampling[<extra>]`` and nothing else, so every member that appears here
arrived through the extra's own published dependency metadata. The extra name arrives
as ``argv[2]``.

The table below *is* the specification: ``present`` names the distributions the extra
declares and must have installed, ``absent`` the kcai members it must not have — the
second half is what makes this a metadata test rather than a smoke test, since a leaked
sibling means the extra declared more than the plan says it does.

Deliberately cheap. No checkpoint download, no torch execution, no real dataset:
distributions are only probed with ``importlib.metadata`` and plugins are read out of
the installed wheel metadata without importing them. Nothing heavy is actually imported
-- notably ``ultralytics``, whose ``cv2`` import needs a system OpenGL library this venv
has no reason to carry. The full walkthrough that *does* run real inference lives behind
``nox -s test_packaging_notebook``.
"""

import importlib.metadata
import pathlib
import sys

UMBRELLA = "kcai-data-sampling"

#: Every kcai member that an extra could conceivably pull.
MEMBERS = {
    "job": "kcai-data-sampling-job",
    "images": "kcai-data-sampling-images",
    "fgsm": "kcai-data-sampling-fgsm",
    "lama": "kcai-data-sampling-lama",
}

#: extra -> (distributions it must install, kcai members it must not).
#:
#: ``notebooks`` carries the third-party notebook stack only. It is the one extra whose
#: "absent" half is load-bearing: dropping ``-fgsm``/``-lama`` from it is exactly what
#: this change did, and nothing else would notice if it silently came back. Note it still
#: resolves torch -- ``ultralytics`` declares torch itself -- which is why the absence
#: claim is about the kcai members, never about torch.
EXTRAS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "notebooks": (
        ("ipykernel", "pandas", "python-pptx", "ultralytics"),
        ("fgsm", "lama"),
    ),
    "job": ((MEMBERS["job"],), ("images", "fgsm", "lama")),
    "images": ((MEMBERS["images"],), ("job", "fgsm", "lama")),
    "fgsm": ((MEMBERS["fgsm"],), ("job", "images", "lama")),
    "lama": ((MEMBERS["lama"],), ("job", "images", "fgsm")),
    "all": ((*MEMBERS.values(),), ()),
}

#: Plugins each member registers, resolved from installed metadata rather than by
#: importing the packages: one representative entry point per member, enough to prove the
#: wheel's own entry points survived publication.
MEMBER_PLUGINS: dict[str, tuple[str, str]] = {
    "job": ("kcai_data_sampling.dataloaders", "parquet"),
    "images": ("kcai_data_sampling.transformations", "horizontal_flip"),
    "fgsm": ("kcai_data_sampling.transformations", "fgsm"),
    "lama": ("kcai_data_sampling.models", "lama_inpaint"),
}


def _installed(distribution: str) -> bool:
    """Whether a distribution is installed, by metadata only (nothing is imported).

    Args:
        distribution: The distribution name to probe.

    Returns:
        True when the distribution resolves.
    """
    try:
        importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _entry_points(group: str) -> dict[str, importlib.metadata.EntryPoint]:
    """The entry points a group resolves to, keyed by name.

    Args:
        group: The entry-point group name.

    Returns:
        A name-keyed mapping of every registered entry point.
    """
    return {ep.name: ep for ep in importlib.metadata.entry_points(group=group)}


def main() -> int:
    scratch = pathlib.Path(sys.argv[1])
    extra = sys.argv[2]
    scratch.mkdir(parents=True, exist_ok=True)

    assert extra in EXTRAS, f"unknown extra {extra!r}"
    present, absent = EXTRAS[extra]

    # The umbrella and its base dependency are always there; the extra is the only thing
    # under test.
    assert _installed(UMBRELLA), f"[{extra}] the umbrella itself is not installed"
    assert _installed("kcai-data-sampling-core"), f"[{extra}] the base core is missing"

    for distribution in present:
        assert _installed(distribution), f"[{extra}] did not install {distribution}"

    for member in absent:
        assert not _installed(MEMBERS[member]), f"[{extra}] unexpectedly pulled - {member}"

    # Every member that did arrive contributes its plugin surface to the registry, which
    # is what a config naming it resolves through.
    for distribution in present:
        member = next((m for m, name in MEMBERS.items() if name == distribution), None)
        if member is None:
            continue
        group, name = MEMBER_PLUGINS[member]
        assert name in _entry_points(group), f"[{extra}] {group} is missing {name!r}"

    # The dispatch reflects exactly the members present: `process` exists only with -job.
    # `present` holds distribution names, so compare against the distribution itself.
    from kcai_data_sampling.dependency import get_available_command

    commands = get_available_command()
    assert {"version", "list"} <= set(commands), sorted(commands)
    assert ("process" in commands) == (MEMBERS["job"] in present), sorted(commands)

    print(f"smoke_extras ok: [{extra}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
