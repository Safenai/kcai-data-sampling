"""Packaging-isolation tests: deselected by default.

Run the suite with ``pytest -m packaging tests/packaging/`` — the default
session (``-m "not packaging"``) leaves these out because each scenario builds
wheels and installs fresh venvs.
"""
