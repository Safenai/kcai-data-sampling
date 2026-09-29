"""The generic ``OutputWriter`` base: a no-op shell a subclass specializes.

The base only records its name; ``add_payload``/``add_rows`` buffer nothing
and ``flush`` writes nothing. The two step-surface writers subclass it, so the
base contract — every call a no-op — is what the plugins inherit.
"""

from kcai_data_sampling_core.api.output_writer import OutputWriter


def test_base_output_writer_stores_its_name() -> None:
    """``name`` is the only state the base keeps; ``config`` is ignored."""
    assert OutputWriter("writer").name == "writer"
    assert OutputWriter("writer", config={"anything": True}).name == "writer"


def test_base_output_writer_adapter_methods_are_noops() -> None:
    """``add_payload``/``add_rows``/``flush`` accept anything and do nothing."""
    writer = OutputWriter("writer")
    assert writer.add_payload("section", object()) is None
    assert writer.add_rows("section", [{"a": 1}]) is None
    assert writer.flush() is None
