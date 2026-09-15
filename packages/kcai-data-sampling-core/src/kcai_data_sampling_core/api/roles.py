"""What a model owes for the role it plays. Any object that satisfies the
protocol will do: a target model is the user's, wrapped in a few lines; a
tool model is chosen by the package (``models/``).

A transformation declares the methods it needs in ``model_methods``; the
base class checks them at construction, and ``check_output`` checks what only a
call can reveal.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TargetModel(Protocol):
    """The model an adversarial transformation is computed *against*."""

    name: str

    def grad(self, xs: np.ndarray) -> np.ndarray:
        """∂loss / ∂x for a batch ``(B, *sample)`` float in [0, 1]; same shape, finite."""
        ...


@runtime_checkable
class ToolModel(Protocol):
    """The model a generative transformation produces content with."""

    name: str

    def inpaint(self, xs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """``(B, *sample)`` and ``(B, H, W)`` bool → ``(B, *sample)``: the masked pixels filled in."""
        ...


def check_model(algorithm: str, role: str, model: Any, methods: tuple[str, ...]) -> None:
    """Refuse, at construction, a model that cannot play the role."""
    name = getattr(model, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{algorithm}: the {role} model must have a non-empty `name`, the row records it")
    missing = [m for m in methods if not callable(getattr(model, m, None))]
    if missing:
        exposes = sorted(m for m in dir(model) if not m.startswith("_") and callable(getattr(model, m)))
        raise ValueError(
            f"{algorithm} needs a {role} model exposing {', '.join(methods)}; "
            f"{name!r} lacks {', '.join(missing)} (it exposes {', '.join(exposes) or 'nothing'})"
        )


def check_output(algorithm: str, model: Any, method: str, xs: np.ndarray, out: Any) -> np.ndarray:
    """Refuse, at the first call, an output that breaks the numeric contract."""
    if not isinstance(out, np.ndarray) or out.shape != xs.shape:
        raise ValueError(
            f"{algorithm}: {model.name}.{method} returned {type(out).__name__}"
            f"{getattr(out, 'shape', '')}, expected an array of shape {xs.shape}"
        )
    if not np.isfinite(out).all():
        raise ValueError(f"{algorithm}: {model.name}.{method} returned non-finite values")
    return out
