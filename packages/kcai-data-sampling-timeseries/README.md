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
- `TimeSeriesBatch` — a `Batch` specialization for windows: axes
  `(time, channel)`, no value range (a signal is unbounded, so outputs are
  neither clipped nor refused on range), and the series' `sampling_rate` when
  declared.

Three methods, as examples of the slot rather than a catalogue.

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
uv run python scripts/make_timeseries_sample.py
uv run kcai-data-sampling process -p examples/config/timeseries-procedural.yaml
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
      seed: 7
```

Listed transformations are applied **independently** to the same source window,
not chained.

## Notes

- **The regime is not declared.** Whether an output is an augmentation or a
  corruption is judged downstream, against a task: the same low-pass keeps
  "is the machine running?" answerable and destroys a high-frequency fault.
- **The window is the sample.** Each window is transformed on its own. With
  overlapping windows, a step shared by two windows can get two different
  outputs, one per window.
- **Floating point only.** Integer input is refused rather than silently
  truncated on the way back.
