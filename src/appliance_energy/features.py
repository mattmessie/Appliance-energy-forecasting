"""Feature engineering for the feature-based ML model (Part 5/6).

Three feature sources, per the assignment brief:

1. Original measured variables: indoor temperature/humidity (T1-T9, RH_1-RH_9)
   and outdoor weather (T_out, Press_mm_hg, RH_out, Windspeed, Visibility,
   Tdewpoint) -- used as-is from the dataset.
2. Time-based features: hour, dayofweek, is_weekend, and their sine/cosine
   encodings -- always known in advance, no leakage risk.
3. Lag and rolling features, built from the target only. All rolling
   features shift(1) BEFORE rolling, so a rolling window ending at time t
   never includes the value at t itself.

Excluded: `lights` (a separate appliance circuit that would itself need
forecasting, like weather -- not in the brief's required covariate list) and
`rv1`/`rv2` (documented in the original UCI dataset release as synthetic
random-noise columns included only to test feature-selection robustness).
"""

import numpy as np
import pandas as pd

EXCLUDED_COLUMNS = ["lights", "rv1", "rv2"]

DEFAULT_LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
DEFAULT_ROLLING_WINDOWS = [3, 6, 12, 24, 168]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour/day-of-week features and their cyclic (sine/cosine) encodings.

    These are always known in advance for any future timestamp, so there is
    no leakage risk in using them for forecasting.
    """
    out = df.copy()

    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    target: str,
    lags: list = None,
    windows: list = None,
) -> pd.DataFrame:
    """Add lag and rolling mean/std features derived from the target.

    Every rolling feature calls `.shift(1)` before `.rolling(...)`, so the
    window ending "at" time t only ever includes values up to and including
    t-1 -- the current/future value of the target is never used to build its
    own predictor.
    """
    if lags is None:
        lags = DEFAULT_LAGS
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS

    out = df.copy()

    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)

    for window in windows:
        shifted = out[target].shift(1)
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std()

    return out


def make_feature_table(
    df: pd.DataFrame,
    target: str,
    lags: list = None,
    windows: list = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build the full supervised-learning feature table.

    Combines: original sensor/weather columns (minus EXCLUDED_COLUMNS),
    time-based features, and target-derived lag/rolling features.
    """
    base = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df.columns])
    out = add_time_features(base)
    out = add_lag_rolling_features(out, target=target, lags=lags, windows=windows)

    if dropna:
        out = out.dropna()

    return out


def feature_columns(feature_table: pd.DataFrame, target: str) -> list:
    """All columns in a feature table except the target itself."""
    return [c for c in feature_table.columns if c != target]
