import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.features import make_feature_table, feature_columns
from appliance_energy.models.feature_models import fit_feature_model, rolling_feature_forecast


def _toy_data(n=250):
    idx = pd.date_range("2016-01-01", periods=n, freq="h")
    rng = np.random.RandomState(0)
    target = 100 + 50 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 3, n)
    weather = 10 + rng.normal(0, 1, n)
    df = pd.DataFrame({"Appliances": target, "T_out": weather}, index=idx)
    return df


def _fit_toy_model(df, lags, windows):
    ft = make_feature_table(df, target="Appliances", lags=lags, windows=windows)
    cols = feature_columns(ft, "Appliances")
    model, _ = fit_feature_model(ft[cols], ft["Appliances"], n_iter=2, cv_splits=2)
    return model, cols


def test_rolling_forecast_covers_full_test_period():
    df = _toy_data(250)
    train, test = df.iloc[:-48], df.iloc[-48:]
    lags, windows = [1, 2, 24], [3, 24]

    model, cols = _fit_toy_model(train, lags, windows)
    fc = rolling_feature_forecast(
        model, df, train["Appliances"], test["Appliances"], cols,
        target="Appliances", lags=lags, windows=windows, block_size=24,
    )

    assert len(fc) == len(test)
    assert list(fc.index) == list(test.index)
    assert fc.notna().all()


def test_rolling_forecast_raises_on_non_multiple_length():
    df = _toy_data(250)
    train, test = df.iloc[:-48], df.iloc[-48:]
    lags, windows = [1, 24], [3]
    model, cols = _fit_toy_model(train, lags, windows)

    with pytest.raises(ValueError):
        rolling_feature_forecast(
            model, df, train["Appliances"], test["Appliances"].iloc[:-1], cols,
            target="Appliances", lags=lags, windows=windows, block_size=24,
        )


def test_rolling_forecast_short_lag_uses_own_prediction_not_future_actual():
    # Regression guard: hour 2 of a block must be built from the model's
    # own hour-1 prediction, not from the real (not-yet-revealed) hour-1
    # actual. We check this indirectly: corrupting the *actual* test values
    # (which the recursive loop should never read for within-block lags)
    # must not change the forecast, since only the model's own predictions
    # feed short lags within a block.
    df = _toy_data(250)
    train, test = df.iloc[:-48], df.iloc[-48:]
    lags, windows = [1, 2], [3]
    model, cols = _fit_toy_model(train, lags, windows)

    fc_a = rolling_feature_forecast(
        model, df, train["Appliances"], test["Appliances"], cols,
        target="Appliances", lags=lags, windows=windows, block_size=24,
    )

    test_corrupted = test.copy()
    test_corrupted["Appliances"] = test_corrupted["Appliances"] + 99999.0
    fc_b = rolling_feature_forecast(
        model, df, train["Appliances"], test_corrupted["Appliances"], cols,
        target="Appliances", lags=lags, windows=windows, block_size=24,
    )

    # Day 1 (first 24 hours) must be identical: nothing in it can depend on
    # ANY test-period actuals (day 1 has no prior revealed test days, and
    # within-day lags use predictions, not actuals).
    pd.testing.assert_series_equal(fc_a.iloc[:24], fc_b.iloc[:24])
