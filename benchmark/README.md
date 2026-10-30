# Benchmark Harness

This harness compares running Albumentations through our data-sampling package against calling Albumentations directly. Same upstream version, same inputs, same machine. The only difference is the loader → transform → clip → `Output` → ledger/payload pipeline.

## Prerequisites

From the repo root:

```bash
UV_INDEX=gitlab-pypi uv sync --frozen
```

## Running benchmarks

The recommended way is via `nox`:

```bash
# Run all scenarios
UV_INDEX=gitlab-pypi uv run --frozen nox -s benchmark

# Run a single scenario
UV_INDEX=gitlab-pypi uv run --frozen nox -s benchmark -- --only s1
UV_INDEX=gitlab-pypi uv run --frozen nox -s benchmark -- --only s2
UV_INDEX=gitlab-pypi uv run --frozen nox -s benchmark -- --only s3
```

You can also run directly without `nox`:

```bash
UV_INDEX=gitlab-pypi uv run --frozen python -m benchmark --only s1
```

## Scenarios

| Scenario | Description |
| --- | --- |
| `s1` | Batch scaling: 1000 small images (128×128), one transform (random brightness/contrast), varying batch sizes (1, 5, 10, 20). |
| `s2` | Transform variety: 40 different transforms on one large image (1928×1208). |
| `s3` | Blur sweep: 40 Gaussian blur levels on one large image (1928×1208). |

## Attribution (phase breakdown)

To get cProfile-based phase breakdowns (for M3-style attribution), run with `--with-attribution`. **Timing will be inflated by cProfile overhead** — use clean runs (without this flag) for headline timing numbers.

```bash
UV_INDEX=gitlab-pypi uv run --frozen nox -s benchmark -- --with-attribution --only s2
UV_INDEX=gitlab-pypi uv run --frozen nox -s benchmark -- --with-attribution --only s3
```

## Outputs

Results are written to:

```
benchmark/outputs/results/<run-id>/
```

Typical artifacts:
- `*_timing.csv` — median/IQR/min/max per cell/side
- `*_memory.csv` — peak RSS per cell/side
- `environment.json` — machine/env metadata
- `equivalence.csv` — parity checks (when applicable)
- `attribution.json` — phase breakdown (only when `--with-attribution` is used)

## Notes

- All numbers are medians of 7 timed repeats per cell/side (warmups applied per runner).
- Equivalence is checked automatically between "ours" and "direct".
- Startup/import time is excluded from timed measurements.
