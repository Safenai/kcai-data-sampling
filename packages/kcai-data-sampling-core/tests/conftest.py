"""Fixtures: the comma10k sample and YOLOv8n, nothing synthetic. Skipped,
not faked, when the sample is not fetched or `ultralytics` is not installed."""

from pathlib import Path

import pytest

from kcai_data_sampling_core import DataSelection, TransformationRunner
from kcai_data_sampling_core.datasets import comma10k

SAMPLE = Path(__file__).resolve().parents[3] / "examples" / "data" / "comma10k_sample"


@pytest.fixture(scope="session")
def samples():
    if not (SAMPLE / "imgs").is_dir():
        pytest.skip("python scripts/fetch_comma10k_sample.py")
    return comma10k.samples(SAMPLE, limit=3)


@pytest.fixture(scope="session")
def target():
    pytest.importorskip("ultralytics", reason="pip install 'kcai-data-sampling-core[yolo]'")
    from kcai_data_sampling_core.models import YoloTarget

    return YoloTarget()


@pytest.fixture(scope="session")
def tool():
    pytest.importorskip("torch", reason="pip install 'kcai-data-sampling-core[lama]'")
    from kcai_data_sampling_core.models import LamaTool

    return LamaTool()


@pytest.fixture
def selection(samples, tmp_path):
    sel = DataSelection("comma10k-three", dataset="comma10k", samples=samples)
    sel.save(tmp_path / "selections")
    return sel


@pytest.fixture
def runner(selection):
    return TransformationRunner(selection)
