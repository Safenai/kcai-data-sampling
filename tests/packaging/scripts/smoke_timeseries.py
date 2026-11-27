"""Time-series smoke: nine registered transforms and one cheap array map.

Runs inside a ``core+timeseries`` venv: metadata proves every algorithm is
registered, and a single ``low_pass`` call proves the wheel's code imports and
runs on a ``(B, W, C)`` window batch. Sibling packages are absent by design, so
the import probes also prove the distribution does not leak them transitively.
"""

import importlib.metadata
import importlib.util
import pathlib
import sys

import numpy as np

EXPECTED_TRANSFORMATIONS = frozenset(
    {
        "low_pass",
        "add_sinusoid",
        "jitter",
        "drift",
        "level_shift",
        "colored_noise",
        "ou_noise",
        "garch_noise",
        "impulse_noise",
    }
)


def _installed(module: str) -> bool:
    """Whether a sibling module root resolves, without importing it."""
    return importlib.util.find_spec(module) is not None


def main() -> int:
    scratch = pathlib.Path(sys.argv[1])
    scratch.mkdir(parents=True, exist_ok=True)

    names = {ep.name for ep in importlib.metadata.entry_points(group="kcai_data_sampling.transformations")}
    assert names >= EXPECTED_TRANSFORMATIONS, sorted(names)
    assert not _installed("kcai_data_sampling_job"), "-job leaked into the timeseries-only venv"
    assert not _installed("kcai_data_sampling_images"), "-images leaked into the timeseries-only venv"

    from kcai_data_sampling_timeseries.api.transformations.low_pass import LowPass

    windows = np.arange(1 * 8 * 2, dtype=np.float32).reshape(1, 8, 2)
    filtered = LowPass({"cutoff": 0.25}).apply(windows)
    assert filtered.shape == windows.shape
    assert np.isfinite(filtered).all()

    print("smoke_timeseries ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
