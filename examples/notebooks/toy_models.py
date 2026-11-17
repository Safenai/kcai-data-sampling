"""Demo models for the newcomer notebook (Air Passengers), on the log scale.

Not part of any package: the simplest forecasts a newcomer can follow, enough
to turn the windows generated in ``toy-demo.ipynb`` into the demo's curves.

- ``seasonal_naive``: next year is this year.
- ``seasonal_growth``: next year is this year, moved by the last yearly growth
  (on the log scale, a growth is an addition).
- ``Detector``: forecasts the last year of a window from the years before; the
  score is how badly it does, and the alert threshold is a quantile of the
  scores on the clean windows.

These models have nothing to fit, so every window can be evaluated; only the
detector's threshold is read from the clean windows.
"""

import numpy as np
import pyarrow.parquet as pq

#: One year, in steps (months).
YEAR = 12


def load_series(path, column="log_values"):
    """Read the only series of a toy sample file.

    Args:
        path: A parquet file written by ``scripts/fetch_toy_samples.py``.
        column: ``log_values`` or ``values``.

    Returns:
        ``(T,)`` float array.
    """
    return np.asarray(pq.read_table(path).to_pylist()[0][column], dtype=float)[:, 0]


def seasonal_naive(histories):
    """Repeat the last year of each history.

    Args:
        histories: ``(N, L)`` array, ``L >= 12``.

    Returns:
        ``(N, 12)`` forecasts.
    """
    return histories[:, -YEAR:]


def seasonal_growth(histories):
    """Repeat the last year, moved by the growth between the last two years.

    Args:
        histories: ``(N, L)`` array on the log scale, ``L >= 24``.

    Returns:
        ``(N, 12)`` forecasts.
    """
    last, previous = histories[:, -YEAR:], histories[:, -2 * YEAR : -YEAR]
    return last + (last.mean(axis=1) - previous.mean(axis=1))[:, None]


class Detector:
    """Window-level detector: how badly the last year of a window is forecast."""

    def __init__(self, quantile=0.95):
        self.quantile = quantile
        self.threshold = None

    @staticmethod
    def score(windows):
        """Mean absolute error of ``seasonal_growth`` on the last year.

        Args:
            windows: ``(N, W)`` array on the log scale, ``W >= 36``.

        Returns:
            ``(N,)`` scores; on the log scale, about a mean relative error.
        """
        return np.abs(seasonal_growth(windows[:, :-YEAR]) - windows[:, -YEAR:]).mean(axis=1)

    def fit(self, clean_windows):
        """Set the threshold from clean windows.

        Args:
            clean_windows: ``(N, W)`` array of windows with no fault.

        Returns:
            The fitted detector.
        """
        self.threshold = float(np.quantile(self.score(clean_windows), self.quantile))
        return self

    def alerts(self, windows):
        """Whether each window raises an alert.

        Args:
            windows: ``(N, W)`` array.

        Returns:
            ``(N,)`` booleans.
        """
        return self.score(windows) > self.threshold
