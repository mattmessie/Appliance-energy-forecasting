import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.features import (
    add_time_features, add_lag_rolling_features, make_feature_table, feature_columns,
)


def _toy_df(n=100):
    idx = pd.date_range("2016-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "Appliances": np.arange(n, dtype=float),
            "lights": np.ones(n),
            "rv1": np.random.RandomState(0).normal(size=n),
            "rv2": np.random.RandomState(1).normal(size=n),
            "T_out": 10 + np.sin(np.arange(n)),
        },
        index=idx,
    )


def test_add_time_features_known_values():
    df = _toy_df(30)
    out = add_time_features(df)
    # 2016-01-01 00:00 is a Friday -> dayofweek 4
    assert out["dayofweek"].iloc[0] == 4
    assert out["hour"].iloc[0] == 0
    assert out["is_weekend"].iloc[0] == 0
    # sine/cosine bounded
    assert out["hour_sin"].between(-1, 1).all()
    assert out["dow_cos"].between(-1, 1).all()


def test_lag_features_do_not_use_future_target_values():
    # Regression guard matching the brief's explicit leakage test example.
    df = _toy_df(50)
    out = add_lag_rolling_features(df, target="Appliances", lags=[1, 2], windows=[3])

    # lag_1 at time t must equal the target at t-1, never t or later.
    shifted = df["Appliances"].shift(1)
    pd.testing.assert_series_equal(out["lag_1"], shifted, check_names=False)


def test_rolling_features_are_shifted_before_rolling():
    df = _toy_df(50)
    out = add_lag_rolling_features(df, target="Appliances", lags=[1], windows=[3])

    # roll_mean_3 at time t should be mean of t-3,t-2,t-1 -- NOT including t.
    expected = df["Appliances"].shift(1).rolling(3).mean()
    pd.testing.assert_series_equal(out["roll_mean_3"], expected, check_names=False)


def test_make_feature_table_excludes_noise_and_lights_columns():
    df = _toy_df(200)
    ft = make_feature_table(df, target="Appliances", lags=[1, 24], windows=[3])
    assert "lights" not in ft.columns
    assert "rv1" not in ft.columns
    assert "rv2" not in ft.columns
    assert "T_out" in ft.columns  # weather kept


def test_make_feature_table_has_no_missing_values_after_dropna():
    df = _toy_df(200)
    ft = make_feature_table(df, target="Appliances", lags=[1, 24, 168], windows=[3, 24])
    assert ft.isna().sum().sum() == 0
    assert len(ft) < len(df)  # rows lost to lag/rolling warm-up


def test_feature_columns_excludes_target():
    df = _toy_df(100)
    ft = make_feature_table(df, target="Appliances", lags=[1], windows=[3])
    cols = feature_columns(ft, "Appliances")
    assert "Appliances" not in cols
