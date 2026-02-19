import os
from pathlib import Path
from typing import Any

import fiftyone.zoo as foz
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ruamel.yaml
import yaml


def get_files_list(path: Path, pattern: str = "*.parquet") -> list[str]:
    files = sorted(Path(path).glob(pattern))
    path_list = [str(x) for x in files]
    return path_list


def plot_histograms(
    output_path: Path,
    columns: list[str],
    parquets_path: Path,
    parquet_pattern: str,
) -> None:
    path_list = get_files_list(parquets_path, pattern=parquet_pattern)

    for parquet_path in path_list:
        parquet_name = Path(parquet_path).stem

        table = pq.read_table(parquet_path)
        for column in columns:
            dist = table[column].to_numpy()

            fig, ax = plt.subplots()

            ax.hist(dist, bins=100)

            plot_name = f"hist_{parquet_name}_{column}.png"

            plt.title(f"Histogram of {column} in {parquet_name}")
            plt.savefig(f"{output_path}/{plot_name}")
            plt.close(fig)


@pytest.fixture(scope="session")
def test_path() -> str:
    # To point on test directory
    return str(Path(__file__).parent.resolve()) + os.sep


@pytest.fixture(scope="session")
def tests_config(test_path: str) -> Any:
    config_path = Path(test_path) / "fixtures" / "expected" / "expected.yaml"

    # Load global unit tests configuration
    with Path.open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    return config


@pytest.fixture(scope="session")
def output_path(test_path: str) -> Path:
    path = Path(test_path) / "outputs" / "data"

    Path.mkdir(path, exist_ok=True, parents=True)

    return path


def write_path_list_to_parquet(path_list: list[Path], save_path: Path) -> None:
    # We ignore type warning from mypy as we really want to convert Path to str
    path_list = [str(x) for x in path_list]  # type: ignore
    path_array = pa.array(path_list)
    path_table = pa.Table.from_arrays([path_array], names=["image_path"])
    pq.write_table(path_table, save_path)

