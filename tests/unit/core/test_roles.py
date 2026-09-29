"""Role conformance: ``check_model`` refuses a model that cannot play its slot.

At construction every role model must carry a non-empty ``name`` (the output
rows record it) and expose the role's required methods; anything short is a
loud ``ValueError`` naming what is missing.
"""

from kcai_data_sampling_core.api.roles import check_model
import pytest


class _NoName:
    pass


class _EmptyName:
    name = ""


class _NamedWithoutMethods:
    name = "named"


class _CompleteTool:
    name = "named"

    def inpaint(self, xs, masks):
        return xs.copy()


def test_check_model_requires_a_non_empty_name() -> None:
    """A name-less object and an empty-name class are both refused."""
    with pytest.raises(ValueError, match=r"non-empty `name`"):
        check_model("t", "tool", _NoName(), ("inpaint",))
    with pytest.raises(ValueError, match=r"non-empty `name`"):
        check_model("t", "tool", _EmptyName(), ("inpaint",))


def test_check_model_refuses_a_model_missing_a_required_method() -> None:
    """A named model without the role's methods is refused, listing them."""
    with pytest.raises(ValueError, match=r"exposing inpaint"):
        check_model("t", "tool", _NamedWithoutMethods(), ("inpaint",))


def test_check_model_accepts_a_fully_conforming_model() -> None:
    """A name plus every required method passes without raising."""
    check_model("t", "tool", _CompleteTool(), ("inpaint",))
