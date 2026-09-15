"""Fixtures: the comma10k sample, read by the core's reader, and a store in a temporary directory."""

from pathlib import Path

import pytest

from kcai_data_sampling_core import DataSelection, TransformationRunner
from kcai_data_sampling_core.datasets import comma10k
from kcai_data_sampling_job import Store

SAMPLE = Path(__file__).resolve().parents[3] / "examples" / "data" / "comma10k_sample"


@pytest.fixture(scope="session")
def samples():
    if not (SAMPLE / "imgs").is_dir():
        pytest.skip("python scripts/fetch_comma10k_sample.py")
    return comma10k.samples(SAMPLE, limit=3)


@pytest.fixture
def selection(samples):
    return DataSelection("comma10k-three", dataset="comma10k", samples=samples)


@pytest.fixture
def runner(selection):
    return TransformationRunner(selection)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "outputs")
