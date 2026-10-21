"""Infrastructure guarantees of the albumentations surface, not per-method.

Four properties that hold once for the whole generated surface: importing the
package never touches the network (the update-guard is set), every spec row
still matches the pinned upstream signature (allowlist rot is caught), the
curated allowlist never slips past the eligibility filter, and no curated name
is claimed by another plugin (registry collisions are refused at test time).
"""

import inspect
import os
import subprocess
import sys

import pytest

pytest.importorskip("kcai_data_sampling_albumentation")

pytestmark = pytest.mark.albumentation

import albumentations as A  # noqa: E402, N812
from kcai_data_sampling_albumentation.specs import TRANSFORM_SPECS  # noqa: E402

#: The hard eligibility filter from the integration plan: classes whose
#: parameter surface or contract breaks the unary image-only, uint8, fixed-shape
#: payload model. Re-checked here so the allowlist cannot silently grow one of
#: these back in.
_INELIGIBLE = {
    # abstract bases
    "DualTransform",
    "ImageOnlyTransform",
    # 3D / volumetric
    "CenterCrop3D",
    "Pad3D",
    "PadIfNeeded3D",
    "RandomCrop3D",
    "CoarseDropout3D",
    "Transform3D",
    # spectrogram aliases
    "TimeMasking",
    "TimeReverse",
    # file-path / reference-image / font-file parameters
    "FDA",
    "HistogramMatching",
    "PixelDistributionAdaptation",
    "TextImage",
    # mask / bbox / keypoints / metadata targets
    "CropNonEmptyMaskIfExists",
    "MaskDropout",
    "ConstrainedCoarseDropout",
    "RandomCropNearBBox",
    "AtLeastOneBBoxRandomCrop",
    "BBoxSafeRandomCrop",
    "RandomSizedBBoxSafeCrop",
    "OverlayElements",
    "Mosaic",
    # value-range / dtype changing
    "Normalize",
    "ToFloat",
    "FromFloat",
    # channel-count changing
    "ToGray",
    "ToRGB",
    # not config-serializable
    "Lambda",
    "NoOp",
    # swaps H x W on non-square images
    "RandomRotate90",
    "Transpose",
}

#: Parameter names that would make a transform dual in disguise.
_TARGET_PARAMETERS = {"mask", "masks", "bboxes", "keypoints", "metadata"}

#: The entry-point group every transformation plugin registers under.
_TRANSFORMATION_GROUP = "kcai_data_sampling.transformations"


def test_import_is_offline_and_sets_the_update_guard() -> None:
    """A fresh interpreter imports the package with sockets blocked.

    The pinned upstream runs a PyPI update check at import unless
    ``NO_ALBUMENTATIONS_UPDATE`` is set; the package must set it before the
    upstream import. The subprocess starts without the variable and makes any
    outbound connection a hard failure, so this pins both halves at once.
    """
    probe = """
import os
import socket

def _blocked(*args, **kwargs):
    raise AssertionError("network access during package import")

socket.socket.connect = _blocked
socket.create_connection = _blocked

import kcai_data_sampling_albumentation.api.transformations  # noqa: F401
import kcai_data_sampling_albumentation.engine  # noqa: F401
print(os.environ.get("NO_ALBUMENTATIONS_UPDATE"))
"""
    env = {key: value for key, value in os.environ.items() if key != "NO_ALBUMENTATIONS_UPDATE"}
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=300)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "1"


def test_specs_match_the_pinned_upstream_signatures() -> None:
    """Every exposed parameter exists upstream with the same default.

    ``p`` is excluded from the default comparison: the decide default is the
    sampling always-on ``1.0``, deliberately overriding upstream's ``0.5``.
    Enum parameters compare through their ``choices`` mapping: the schema pins
    the friendly name (``bilinear``), upstream stores the value it maps to
    (``1``) — so an upstream code change still breaks the row. Pinned-only
    kwargs must also exist, so a pinned shape knob cannot silently disappear
    from the frozen upstream.
    """
    for type_name, spec in TRANSFORM_SPECS.items():
        upstream = getattr(A, spec.source)
        parameters = inspect.signature(upstream).parameters
        for name, param in spec.params.items():
            assert name in parameters, f"{type_name}: {spec.source} has no parameter {name!r}"
            if name == "p":
                continue
            if param.kind == "enum":
                assert param.choices is not None
                assert param.choices[param.default] == parameters[name].default, (
                    f"{type_name}: {name} default {param.default!r} no longer maps to upstream "
                    f"{parameters[name].default!r}"
                )
            elif param.default is ...:
                # Upstream marks a required parameter with ``inspect._empty``.
                assert parameters[name].default is inspect.Parameter.empty, (
                    f"{type_name}: {name} is no longer required upstream "
                    f"(default {parameters[name].default!r}); give the spec row a default"
                )
            else:
                assert parameters[name].default == param.default, (
                    f"{type_name}: {name} default drifted from {param.default!r}"
                )
        for name in spec.pinned:
            assert name in parameters, f"{type_name}: {spec.source} has no pinned parameter {name!r}"


def test_specs_respect_the_eligibility_filter() -> None:
    """No curated row is an ineligible class, a dual target, or a callable knob.

    A target parameter disqualifies a transform only when it is required or
    defaults to something other than ``None``: an optional ``mask=None``
    (``equalize``) never participates — the unary engine never passes it, and
    the spec table never exposes one (asserted directly below).
    """
    sources = {spec.source for spec in TRANSFORM_SPECS.values()}
    assert not sources & _INELIGIBLE, f"ineligible transforms in the allowlist: {sources & _INELIGIBLE}"

    for type_name, spec in TRANSFORM_SPECS.items():
        exposed_targets = set(spec.params) & _TARGET_PARAMETERS
        assert not exposed_targets, f"{type_name}: exposed target parameters {exposed_targets}"
        for name, parameter in inspect.signature(getattr(A, spec.source)).parameters.items():
            if parameter.default is not None:
                assert name not in _TARGET_PARAMETERS, (
                    f"{type_name}: target parameter {name!r} makes it a dual transform"
                )
            # A required parameter has the ``inspect._empty`` sentinel as its
            # default, which is callable but not a user-supplied callable.
            if parameter.default is not inspect.Parameter.empty:
                assert not callable(parameter.default), (
                    f"{type_name}: callable parameter {name!r} is not config-serializable"
                )


def test_curated_names_are_not_claimed_by_another_plugin() -> None:
    """Each curated type resolves to exactly one entry point, and it is ours."""
    from importlib.metadata import entry_points

    claims: dict[str, list[str]] = {}
    for entry_point in entry_points(group=_TRANSFORMATION_GROUP):
        claims.setdefault(entry_point.name, []).append(entry_point.value)

    for type_name in TRANSFORM_SPECS:
        values = claims.get(type_name, [])
        assert len(values) == 1, f"{type_name}: expected exactly one entry point, found {values}"
        assert values[0].startswith("kcai_data_sampling_albumentation."), (
            f"{type_name}: claimed by another package: {values[0]}"
        )
