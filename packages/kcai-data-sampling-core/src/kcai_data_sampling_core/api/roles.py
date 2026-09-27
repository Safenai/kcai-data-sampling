"""The roles a model can play, checked as protocols.

Any object that satisfies the protocol will do: a target model is the user's,
wrapped in a few lines; a tool model is chosen by an algorithm package
(``models/``). A transformation declares the methods it needs in
``model_methods``; the base class checks them at construction, and
``check_output`` checks what only a call can reveal.
"""

import numpy as np
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TargetModel(Protocol):
    """The model an adversarial transformation is computed *against*.

    Attributes:
        name: Readable name, recorded on every output row.
    """

    name: str

    def grad(self, xs: np.ndarray) -> np.ndarray:
        """Differentiate the loss with respect to the input.

        The input is the normalized float batch in ``[0, 1]`` — an integer
        batch mapped by dividing through by its dtype maximum. The returned
        gradient is in the same normalized units; only its sign participates
        in the perturbation step.

        Args:
            xs: Input batch ``(B, *sample)`` normalized to ``[0, 1]``.

        Returns:
            The gradient ``∂loss / ∂x`` for the batch, same shape, finite.
        """
        ...


@runtime_checkable
class ToolModel(Protocol):
    """The model a generative transformation produces content with.

    Attributes:
        name: Readable name, recorded on every output row.
    """

    name: str

    def inpaint(self, xs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Fill the masked pixels of a batch.

        Args:
            xs: Input batch ``(B, *sample)``.
            masks: One boolean mask per sample ``(B, H, W)``.

        Returns:
            The batch ``(B, *sample)`` with the masked pixels filled in.
        """
        ...


def check_model(algorithm: str, role: str, model: Any, methods: tuple[str, ...]) -> None:
    """Refuse, at construction, a model that cannot play the role it is given.

    Args:
        algorithm: The transformation's identity, for the error message.
        role: ``"tool"`` or ``"target"``, the slot the model must fill.
        model: The candidate model instance.
        methods: The methods the role's protocol requires.

    Raises:
        ValueError: If the model has no non-empty ``name`` or lacks a required method.
    """
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
    """Refuse, at the first call, an output that breaks the numeric contract.

    Args:
        algorithm: The transformation's identity, for the error message.
        model: The model that produced the output.
        method: The method name that was called, for the error message.
        xs: The input batch the output must match in shape.
        out: Whatever the model returned.

    Returns:
        The output, proven to be a finite array of the expected shape.

    Raises:
        ValueError: If the output is not a numpy array of the expected shape
            or contains non-finite values.
    """
    if not isinstance(out, np.ndarray) or out.shape != xs.shape:
        raise ValueError(
            f"{algorithm}: {model.name}.{method} returned {type(out).__name__}"
            f"{getattr(out, 'shape', '')}, expected an array of shape {xs.shape}"
        )
    if not np.isfinite(out).all():
        raise ValueError(f"{algorithm}: {model.name}.{method} returned non-finite values")
    return out