from pathlib import Path
import yaml
from timeit import default_timer as timer
from typing import Any

import pytest

from kcai_data_sampling import RunAugmentPipeline


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "test_name",
    ["brightness"],
)
def test_augment(
    tests_config: Any,
    test_path: Path,
    output_path: Path,
    test_name: str,
) -> None:
    # pad and cmd not implemented

    with open(f"examples/augment_{test_name}.yml", "r") as augment_file:
        augment_config = yaml.safe_load(augment_file)
        start = timer()
        # Try to load as TorchScript first (for .pt files), otherwise as regular checkpoint

        # Force TorchScript quantized models to CPU
        RunAugmentPipeline(augment_config)

        end = timer()
    print(f"Execution time: {end - start}")
