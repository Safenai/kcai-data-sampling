# kcai-data-sampling-timeseries

The **procedural** time-series slot of kcai data-sampling: transformations whose
output is computed from the input window alone. No model enters the
computation, so nothing here needs a framework and nothing here needs a GPU.

The sample is a **window**: `W` consecutive steps of a series over `C`
channels. These methods work in memory on a `(B, W, C)` floating-point stack,
where `B` is the batch and axis 1 is time. They return a batch of the same
shape and dtype — the sample space never changes, so the difference between a
source window and its output stays defined and can be computed downstream.

Its only dependencies are numpy and the core contracts. It deliberately ships no
input/output plugins: cutting series into windows and writing the outputs are
handled by the job package.

## What it ships today

- `low_pass` — keep the frequencies at or below `cutoff`, zero the others (ideal
  FFT filter along time). Deterministic, not reversible. Global support: the
  output at one step depends on the whole window.
- `add_sinusoid` — add `amplitude * sin(2π * frequency * t + phase)` to every
  channel, `t` counted from the window's first step. Deterministic, not
  reversible. Pointwise support.
- `jitter` — add independent Gaussian noise of standard deviation `sigma`
  (signal units) to every step and channel. Stochastic: requires a `seed`;
  each window draws from its own generator, seeded by the seed, its parent id
  and the parameters, so its output never depends on the batch. Not
  reversible. Pointwise support.
- `drift` — add a linear ramp `slope * t` to every channel, `t` counted from
  the window's first step, so the first step is unchanged; `slope` is per step,
  in signal units, any sign. Deterministic, not reversible. Pointwise support.
- `level_shift` — add `offset` to every channel from the step at `start` on,
  `start` being a fraction of the window (default 0.5): an abrupt break that
  stays, where `drift` is gradual. Deterministic, not reversible. Pointwise
  support.
- Four **time-series noises**, correlated in time unlike `jitter`. Each is
  stochastic (requires a `seed`, one generator per window), not reversible,
  and takes a `mode`: `additive` (`x + n`, default, noise in signal units) or
  `multiplicative` (`x * (1 + n)`, noise as a relative change that scales with
  the signal).
  - `colored_noise` — noise whose power spectrum falls as `1/f^exponent`
    (`exponent` 0 white, 1 pink, 2 brown), rescaled to standard deviation
    `sigma` over each window. Global support.
  - `ou_noise` — Ornstein-Uhlenbeck: a mean-reverting random walk of
    stationary deviation `sigma`, pulled back at rate `theta` per step
    (correlated over about `1 / theta` steps). Causal support.
  - `garch_noise` — GARCH(1,1): volatility that clusters in time,
    unconditional deviation `sigma`, shock weight `alpha`, persistence `beta`,
    `alpha + beta < 1`. Causal support.
  - `impulse_noise` — isolated spikes ("salt and pepper"): each step is hit
    with probability `rate` by `±amplitude`. Pointwise support.
- `TimeSeriesBatch` — a `Batch` specialization for windows: axes
  `(time, channel)`, no value range (a signal is unbounded, so outputs are
  neither clipped nor refused on range), and the series' `sampling_rate` when
  declared.

Nine methods, as examples of the slot rather than a catalogue.

Every method transforms **every channel** of the windows it receives, with the
same parameters. Which channels a window holds is decided upstream, by the data
selection: the `timeseries_parquet` loader's `channels` keeps the listed ones.
Parameters are in the units of the signal, so select the channels whose units
fit them.

## Frequencies are in cycles per step

`cutoff` and `frequency` are given in **cycles per step**, in `(0, 0.5]`, where
`0.5` is the Nyquist frequency. A transformation receives the window array
only, not the series' sampling rate, so it cannot read a frequency in hertz.
For a series sampled at `fs` Hz, `f` Hz is `f / fs` cycles per step: 5 Hz at
100 Hz is `0.05`.

A sinusoid at exactly `0.5` with `phase: 0` is zero at every step: pick a
frequency below Nyquist, or a non-zero phase.

## Install

Not published yet, and not part of the umbrella's extras: the package is
available from a checkout of the repository, where `uv sync` installs it with
the rest of the workspace.

```bash
uv run python scripts/fetch_ett_sample.py
uv run kcai-data-sampling process -p examples/config/ett-procedural.yaml
```

Cutting series into windows (the `timeseries_parquet` loader) and writing the
`.npy` outputs belong to `kcai-data-sampling-job`.

## Example

```yaml
operations:
  transformations:
    - name: low_pass
      type: low_pass
      cutoff:
        range: [0.05, 0.2]
        samples: 3
        mode: even
    - name: interference
      type: add_sinusoid
      frequency: 0.25
      amplitude: 0.1
    - name: sensor_noise
      type: jitter
      sigma: 0.05
      seed: 0
```

Listed transformations are applied **independently** to the same source window,
not chained.

## Demo on real data (ETT)

[ETT](https://github.com/zhouhaoyi/ETDataset) (*Electricity Transformer
Temperature*) is the reference forecasting benchmark: hourly load and oil
temperature of power transformers. The demo treats the oil temperature as a
transformer operator would, with two use cases:

- **forecasting**: degraded histories (`jitter`, `low_pass`) measure how fast
  a forecaster's error grows (`examples/config/ett-forecasting.yaml`);
- **anomaly detection**: harmless variability (`jitter`, `low_pass`, and the
  `colored_noise`, `ou_noise` and `garch_noise` noises swept from mild to
  strong) and injected faults (`drift` for cooling degradation, `add_sinusoid`
  for cooling cycling, `impulse_noise` for acquisition glitches) measure a
  detector's false alarms, how much variability it tolerates and the smallest
  fault it catches (`examples/config/ett-anomaly.yaml`). The label of each generated window is
  read from the ledger.

```bash
uv run python scripts/fetch_ett_sample.py
uv run kcai-data-sampling process -p examples/config/ett-anomaly.yaml
uv run kcai-data-sampling process -p examples/config/ett-forecasting.yaml
```

`examples/notebooks/ett-demo.ipynb` is the same demo as a tutorial: it cuts
the windows with the loader, calls the transformation classes and sweeps their
parameters itself, evaluates simple stand-in models from
`examples/notebooks/ett_models.py`, then runs the three ETT configs and checks
that the pipeline writes the same windows (it needs scikit-learn and
matplotlib, which are not workspace dependencies). `air-passengers-demo.ipynb` does the
same on Air Passengers, for newcomers.

ETT is licensed CC BY-ND 4.0: the data is downloaded by the script, never
shipped, and the generated windows and figures are not to be committed or
shared.

## Notes

- **The regime is not declared.** Whether an output is an augmentation or a
  corruption is judged downstream, against a task: the same low-pass keeps
  "is the machine running?" answerable and destroys a high-frequency fault.
- **The window is the sample.** Each window is transformed on its own. With
  overlapping windows, a step shared by two windows can get two different
  outputs, one per window.
- **Floating point only.** Integer input is refused rather than silently
  truncated on the way back.
