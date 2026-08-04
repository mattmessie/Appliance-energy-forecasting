"""Simple benchmark forecasts: mean, naive, seasonal naive (daily/weekly),
and drift.

Two layers are provided:

- The single-origin functions (`mean_forecast`, `naive_forecast`, etc.)
  compute a forecast for `test_index` using only `train`. They know nothing
  about rolling origins and are unit-tested in isolation.
- `generate_rolling_benchmarks` wraps these into the assignment's actual
  evaluation design (see report.md, Section 4): a 24-hour-ahead forecast,
  repeated at 14 daily origins across the test period, where each origin's
  "train" is an expanding history (initial training set + every test day
  already revealed by that point). No forecast ever uses a value from later
  in the test period than its own origin.
"""

import numpy as np
import pandas as pd


def mean_forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Flat forecast at the training-set mean for every step of the horizon."""
    value = train.mean()
    return pd.Series(value, index=test_index, name="mean")


def naive_forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Flat forecast at the last observed training value (persistence)."""
    value = train.iloc[-1]
    return pd.Series(value, index=test_index, name="naive")


def _seasonal_naive(
    train: pd.Series, test_index: pd.DatetimeIndex, period: int, name: str
) -> pd.Series:
    """Tile the last full seasonal cycle of the training set across the
    whole forecast horizon (repeats if horizon > period).
    """
    if len(train) < period:
        raise ValueError(
            f"Training series ({len(train)} obs) shorter than period ({period})."
        )
    last_cycle = train.iloc[-period:].values
    horizon = len(test_index)
    n_tiles = int(np.ceil(horizon / period))
    tiled = np.tile(last_cycle, n_tiles)[:horizon]
    return pd.Series(tiled, index=test_index, name=name)


def seasonal_naive_daily_forecast(
    train: pd.Series, test_index: pd.DatetimeIndex, period: int = 24
) -> pd.Series:
    """Repeats the last observed 24-hour cycle from training across the
    whole test horizon: forecast for hour h of day d = value at the same
    hour-of-day in the last training day.
    """
    return _seasonal_naive(train, test_index, period, "seasonal_naive_daily")


def seasonal_naive_weekly_forecast(
    train: pd.Series, test_index: pd.DatetimeIndex, period: int = 168
) -> pd.Series:
    """Repeats the last observed 168-hour (7-day) cycle from training
    across the whole test horizon.
    """
    return _seasonal_naive(train, test_index, period, "seasonal_naive_weekly")


def drift_forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Random walk with drift: extrapolates the average slope observed
    across the whole training set from the last training value.

    forecast(h) = y_T + h * (y_T - y_1) / (T - 1),  h = 1, ..., H
    """
    y_first = train.iloc[0]
    y_last = train.iloc[-1]
    n_train = len(train)
    if n_train < 2:
        raise ValueError("Need at least 2 training observations for drift.")
    slope = (y_last - y_first) / (n_train - 1)
    horizon = len(test_index)
    steps = np.arange(1, horizon + 1)
    values = y_last + steps * slope
    return pd.Series(values, index=test_index, name="drift")


def generate_all_benchmarks(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    daily_period: int = 24,
    weekly_period: int = 168,
) -> dict:
    """Run all five benchmark models from a single origin and return
    {name: forecast_series}. Building block for `generate_rolling_benchmarks`;
    also usable directly for a plain single-origin forecast of any horizon.
    """
    return {
        "mean": mean_forecast(train, test_index),
        "naive": naive_forecast(train, test_index),
        "seasonal_naive_daily": seasonal_naive_daily_forecast(
            train, test_index, daily_period
        ),
        "seasonal_naive_weekly": seasonal_naive_weekly_forecast(
            train, test_index, weekly_period
        ),
        "drift": drift_forecast(train, test_index),
    }


def generate_rolling_benchmarks(
    train: pd.Series,
    test: pd.Series,
    block_size: int = 24,
    daily_period: int = 24,
    weekly_period: int = 168,
) -> dict:
    """Rolling walk-forward version of `generate_all_benchmarks`.

    Splits `test` into consecutive blocks of `block_size` hours (default 24,
    i.e. one day). For each block, all five benchmarks are computed using
    only the initial training set plus whatever test blocks came before it
    (an expanding history) — never any value from the block being forecast
    or later. The per-block forecasts are concatenated into a single series
    per model spanning the whole test period.

    Parameters
    ----------
    train : pd.Series
        Initial training series (before the test period starts).
    test : pd.Series
        Full test period, e.g. 336 hourly values (14 days). Its length must
        be an exact multiple of `block_size`.
    block_size : int
        Forecast horizon per rolling origin, in hours (24 = one day, per the
        assignment brief).

    Returns
    -------
    dict[str, pd.Series] mapping model name -> forecast series covering the
    whole test period.
    """
    if len(test) % block_size != 0:
        raise ValueError(
            f"Test period length ({len(test)}) is not a multiple of "
            f"block_size ({block_size})."
        )

    n_blocks = len(test) // block_size
    model_names = ["mean", "naive", "seasonal_naive_daily", "seasonal_naive_weekly", "drift"]
    parts = {name: [] for name in model_names}

    for k in range(n_blocks):
        block_index = test.index[k * block_size : (k + 1) * block_size]
        already_revealed = test[test.index < block_index[0]]
        history = pd.concat([train, already_revealed])

        block_forecasts = generate_all_benchmarks(
            history, block_index, daily_period=daily_period, weekly_period=weekly_period
        )
        for name in model_names:
            parts[name].append(block_forecasts[name])

    return {name: pd.concat(series_list) for name, series_list in parts.items()}
