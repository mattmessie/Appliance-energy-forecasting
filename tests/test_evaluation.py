import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.evaluation import mae, rmse, bias, mase, evaluate_all


def _toy_series():
    idx_train = pd.date_range("2016-01-01", periods=50, freq="h")
    train = pd.Series(np.linspace(10, 20, 50), index=idx_train)

    idx_test = pd.date_range(idx_train[-1] + pd.Timedelta(hours=1), periods=10, freq="h")
    test = pd.Series(np.linspace(20, 25, 10), index=idx_test)
    return train, test


def test_mase_zero_for_perfect_forecast():
    train, test = _toy_series()
    perfect_forecast = test.copy()
    assert mase(test, perfect_forecast, train) == pytest.approx(0.0)


def test_mae_rmse_bias_basic():
    actual = pd.Series([10.0, 20.0, 30.0])
    forecast = pd.Series([12.0, 18.0, 33.0])

    assert mae(actual, forecast) == pytest.approx((2 + 2 + 3) / 3)
    assert rmse(actual, forecast) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    # bias = mean(forecast - actual) = mean(2, -2, 3) = 1
    assert bias(actual, forecast) == pytest.approx(1.0)


def test_evaluate_all_returns_expected_columns_and_sorted_by_mase():
    train, test = _toy_series()
    forecasts = {
        "perfect": test.copy(),
        "flat_zero": pd.Series(0.0, index=test.index),
    }
    result = evaluate_all(test, forecasts, train)

    assert list(result.columns) == ["model", "MAE", "RMSE", "MASE", "Bias"]
    assert len(result) == 2
    # perfect forecast should rank first (lowest MASE)
    assert result.iloc[0]["model"] == "perfect"
    assert result.iloc[0]["MASE"] == pytest.approx(0.0)


def test_evaluate_all_raises_on_missing_values_after_alignment():
    train, test = _toy_series()
    bad_index = test.index[:-2]  # shorter than test -> reindex introduces NaN
    forecasts = {"bad": test.loc[bad_index]}

    with pytest.raises(ValueError):
        evaluate_all(test, forecasts, train)
