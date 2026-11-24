"""Demo models for the ETT notebook: a forecaster and an anomaly detector.

Not part of any package: these are deliberately simple stand-ins for the
operator's models, enough to turn the windows generated in ``timeseries-walkthrough.ipynb``
into the two curves of the demo (forecast error vs. degradation, detection
rate vs. fault size).

- ``SeasonalNaive``: tomorrow is today (the last 24 hours repeated).
- ``LagRidge``: a ridge regression from the history to the next ``horizon``
  hours, on values relative to the last observed one.
- ``SurpriseDetector``: a ``LagRidge`` that predicts the last ``horizon`` hours
  of a window from the rest; the anomaly score is how badly it does. The alert
  threshold is a quantile of the scores on clean training windows, so the
  false-alarm rate is set by construction.

Everything is fitted on the first ``TRAIN_FRACTION`` of each series and
evaluated on windows that start after it, so no evaluated window overlaps the
training data.

Needs scikit-learn, which is not a workspace dependency:
``uv run --with scikit-learn …``.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge

#: Index of the oil temperature in the ETT channels.
OT = 6
#: Share of each series used for fitting; evaluation starts after it.
TRAIN_FRACTION = 0.7
#: One day, in steps (hours).
DAY = 24


def load_ot(path):
    """Read the oil temperature of every series of the ETT sample.

    Args:
        path: The ``series.parquet`` written by ``scripts/fetch_ett_sample.py``.

    Returns:
        ``{series_id: (T,) float array}``.
    """
    table = pq.read_table(path).to_pandas()
    return {r.series_id: np.asarray(r["values"].tolist(), dtype=float)[:, OT] for _, r in table.iterrows()}


def train_end(series):
    """First step that is not training data, per series.

    Args:
        series: ``{series_id: (T,) array}``.

    Returns:
        ``{series_id: int}``.
    """
    return {k: int(len(v) * TRAIN_FRACTION) for k, v in series.items()}


def cut(x, length, stop, stride=1):
    """All windows of ``length`` steps of ``x`` that end before ``stop``.

    Args:
        x: ``(T,)`` series.
        length: Window length in steps.
        stop: Windows end strictly before this step.
        stride: Steps between two window starts.

    Returns:
        ``(N, length)`` array.
    """
    starts = range(0, stop - length + 1, stride)
    return np.stack([x[s : s + length] for s in starts])


class SeasonalNaive:
    """Forecast the next ``horizon`` hours as a copy of the last day."""

    def __init__(self, horizon=DAY):
        self.horizon = horizon

    def fit(self, histories, futures):
        """Nothing to fit; kept for a uniform interface."""
        return self

    def predict(self, histories):
        """Repeat the last day of each history.

        Args:
            histories: ``(N, L)`` array, ``L >= 24``.

        Returns:
            ``(N, horizon)`` forecasts.
        """
        reps = -(-self.horizon // DAY)
        return np.tile(histories[:, -DAY:], reps)[:, : self.horizon]


class LagRidge:
    """Ridge regression from a history to the next ``horizon`` hours.

    Inputs and targets are taken relative to the last observed value, so one
    model serves transformers at different temperature levels.
    """

    def __init__(self, horizon=DAY, alpha=10.0):
        self.horizon = horizon
        self.model = Ridge(alpha=alpha)

    def fit(self, histories, futures):
        """Fit on clean windows.

        Args:
            histories: ``(N, L)`` array.
            futures: ``(N, horizon)`` array.

        Returns:
            The fitted model.
        """
        last = histories[:, -1:]
        self.model.fit(histories - last, futures - last)
        return self

    def predict(self, histories):
        """Forecast each history.

        Args:
            histories: ``(N, L)`` array, same ``L`` as at fit time.

        Returns:
            ``(N, horizon)`` forecasts.
        """
        last = histories[:, -1:]
        return self.model.predict(histories - last) + last


def training_set(series, history, horizon=DAY):
    """Clean (history, future) pairs from the training part of every series.

    Args:
        series: ``{series_id: (T,) array}``.
        history: History length in steps.
        horizon: Forecast length in steps.

    Returns:
        ``(histories (N, history), futures (N, horizon))``.
    """
    ends = train_end(series)
    pairs = np.concatenate([cut(x, history + horizon, ends[k]) for k, x in series.items()])
    return pairs[:, :history], pairs[:, history:]


class SurpriseDetector:
    """Window-level anomaly detector: how badly the end of a window is predicted.

    The last ``horizon`` steps of a window are forecast from the steps before
    them; the score is the mean absolute error of that forecast. An alert is
    raised above the ``quantile`` of the scores on clean training windows.
    """

    def __init__(self, window, horizon=DAY, quantile=0.95):
        self.window = window
        self.horizon = horizon
        self.quantile = quantile
        self.forecaster = LagRidge(horizon=horizon)
        self.threshold = None

    def score(self, windows):
        """Anomaly score of each window.

        Args:
            windows: ``(N, window)`` array.

        Returns:
            ``(N,)`` mean absolute error on the last ``horizon`` steps.
        """
        head, tail = windows[:, : -self.horizon], windows[:, -self.horizon :]
        return np.abs(self.forecaster.predict(head) - tail).mean(axis=1)

    def fit(self, series):
        """Fit the forecaster and the threshold on clean training windows.

        Args:
            series: ``{series_id: (T,) array}``.

        Returns:
            The fitted detector.
        """
        heads, tails = training_set(series, self.window - self.horizon, self.horizon)
        self.forecaster.fit(heads, tails)
        self.threshold = float(np.quantile(self.score(np.concatenate([heads, tails], axis=1)), self.quantile))
        return self

    def alerts(self, windows):
        """Whether each window raises an alert.

        Args:
            windows: ``(N, window)`` array.

        Returns:
            ``(N,)`` booleans.
        """
        return self.score(windows) > self.threshold


def forecast_errors(ledger, windows, series, models, horizon=DAY):
    """Forecast error of each model on each degraded history.

    The true future is read from the original series after ``end_window``, so
    it is never transformed. Rows whose future runs past the series are
    dropped.

    Args:
        ledger: One row per generated history, with ``series_id``,
            ``end_window``, ``algorithm`` and ``value`` (the swept
            parameter).
        windows: Their degraded histories, ``(N, W)``.
        series: ``{series_id: (T,) array}``.
        models: ``{name: fitted forecaster}``.
        horizon: Forecast length in steps.

    Returns:
        One row per (ledger row, model): ``algorithm``, ``value``, ``model``,
        ``mae``.
    """
    keep = np.array([e + horizon < len(series[s]) for s, e in zip(ledger.series_id, ledger.end_window, strict=True)])
    rows, hist = ledger[keep].reset_index(drop=True), windows[keep]
    truth = np.stack([series[s][e + 1 : e + 1 + horizon] for s, e in zip(rows.series_id, rows.end_window, strict=True)])
    frames = []
    for name, model in models.items():
        mae = np.abs(model.predict(hist) - truth).mean(axis=1)
        frames.append(pd.DataFrame({"algorithm": rows.algorithm, "value": rows.value, "model": name, "mae": mae}))
    return pd.concat(frames, ignore_index=True)
