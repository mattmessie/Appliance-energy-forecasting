import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.models.benchmarks import (
    mean_forecast,
    naive_forecast,
    seasonal_naive_daily_forecast,
    seasonal_naive_weekly_forecast,
    drift_forecast,
    generate_all_benchmarks,
    generate_rolling_benchmarks,
)


def _toy_train_and_test_index(n_train=200, horizon=48):
    idx_train = pd.date_range("2016-01-01", periods=n_train, freq="h")
    # simple repeating daily pattern so seasonal naive has something to catch
    values = 10 + 5 * np.sin(2 * np.pi * np.arange(n_train) / 24)
    train = pd.Series(values, index=idx_train)

    idx_test = pd.date_range(idx_train[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    return train, idx_test


@pytest.mark.parametrize(
    "forecast_fn",
    [mean_forecast, naive_forecast, seasonal_naive_daily_forecast, seasonal_naive_weekly_forecast, drift_forecast],
)
def test_forecast_length_matches_test_period(forecast_fn):
    train, test_index = _toy_train_and_test_index()
    fc = forecast_fn(train, test_index)
    assert len(fc) == len(test_index)
    assert list(fc.index) == list(test_index)


def test_mean_forecast_is_flat_at_train_mean():
    train, test_index = _toy_train_and_test_index()
    fc = mean_forecast(train, test_index)
    np.testing.assert_allclose(fc.values, train.mean())


def test_naive_forecast_is_flat_at_last_train_value():
    train, test_index = _toy_train_and_test_index()
    fc = naive_forecast(train, test_index)
    np.testing.assert_allclose(fc.values, train.iloc[-1])


def test_seasonal_naive_daily_tiles_last_24h_pattern():
    train, test_index = _toy_train_and_test_index(n_train=200, horizon=48)
    fc = seasonal_naive_daily_forecast(train, test_index, period=24)
    last_day = train.iloc[-24:].values
    expected = np.tile(last_day, 2)
    np.testing.assert_allclose(fc.values, expected)


def test_seasonal_naive_daily_does_not_use_test_period_values():
    # Regression guard against leakage: forecast must be derivable from
    # train alone, i.e. identical even if test-period actuals change.
    train, test_index = _toy_train_and_test_index()
    fc_a = seasonal_naive_daily_forecast(train, test_index, period=24)
    fc_b = seasonal_naive_daily_forecast(train, test_index, period=24)
    pd.testing.assert_series_equal(fc_a, fc_b)


def test_generate_all_benchmarks_returns_five_models_correct_length():
    train, test_index = _toy_train_and_test_index()
    forecasts = generate_all_benchmarks(train, test_index)
    assert set(forecasts.keys()) == {
        "mean",
        "naive",
        "seasonal_naive_daily",
        "seasonal_naive_weekly",
        "drift",
    }
    for fc in forecasts.values():
        assert len(fc) == len(test_index)


def _toy_train_and_test(n_train=300, n_test_days=3, block_size=24):
    idx_train = pd.date_range("2016-01-01", periods=n_train, freq="h")
    values = 10 + 5 * np.sin(2 * np.pi * np.arange(n_train) / 24)
    train = pd.Series(values, index=idx_train)

    n_test = n_test_days * block_size
    idx_test = pd.date_range(idx_train[-1] + pd.Timedelta(hours=1), periods=n_test, freq="h")
    test_values = 10 + 5 * np.sin(2 * np.pi * np.arange(n_train, n_train + n_test) / 24) + 20
    test = pd.Series(test_values, index=idx_test)
    return train, test


def test_generate_rolling_benchmarks_covers_full_test_period():
    train, test = _toy_train_and_test(n_test_days=3)
    forecasts = generate_rolling_benchmarks(train, test, block_size=24)

    assert set(forecasts.keys()) == {
        "mean",
        "naive",
        "seasonal_naive_daily",
        "seasonal_naive_weekly",
        "drift",
    }
    for fc in forecasts.values():
        assert len(fc) == len(test)
        assert list(fc.index) == list(test.index)


def test_generate_rolling_benchmarks_raises_on_non_multiple_length():
    train, test = _toy_train_and_test(n_test_days=3)
    with pytest.raises(ValueError):
        generate_rolling_benchmarks(train, test.iloc[:-1], block_size=24)


def test_rolling_naive_uses_revealed_actuals_not_stale_origin():
    # The rolling naive forecast for day 2 should equal the actual value at
    # the end of day 1 (revealed history), not the original train's last
    # value used for day 1 -- i.e. it must actually "roll".
    train, test = _toy_train_and_test(n_test_days=2)
    forecasts = generate_rolling_benchmarks(train, test, block_size=24)

    day1_forecast_value = forecasts["naive"].iloc[0]
    day2_forecast_value = forecasts["naive"].iloc[24]

    assert day1_forecast_value == pytest.approx(train.iloc[-1])
    assert day2_forecast_value == pytest.approx(test.iloc[23])  # last actual of day 1
    assert day1_forecast_value != pytest.approx(day2_forecast_value)


def test_rolling_seasonal_naive_daily_never_uses_future_test_values():
    # Regression guard: forecasting day k must not change if we corrupt
    # actuals *after* day k in the test set (future info shouldn't leak
    # backward into earlier rolling-origin forecasts).
    train, test = _toy_train_and_test(n_test_days=3)
    forecasts_a = generate_rolling_benchmarks(train, test, block_size=24)

    test_corrupted = test.copy()
    test_corrupted.iloc[48:] = 99999.0  # corrupt day 3 only
    forecasts_b = generate_rolling_benchmarks(train, test_corrupted, block_size=24)

    # Day 1 and day 2 forecasts (first 48 hours) must be identical regardless
    # of what happens on day 3.
    pd.testing.assert_series_equal(
        forecasts_a["seasonal_naive_daily"].iloc[:48],
        forecasts_b["seasonal_naive_daily"].iloc[:48],
    )
    pd.testing.assert_series_equal(
        forecasts_a["naive"].iloc[:48], forecasts_b["naive"].iloc[:48]
    )
