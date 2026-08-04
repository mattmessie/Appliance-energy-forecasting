import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.models.sarimax import fit_sarimax, rolling_sarimax_forecast


def _toy_series(n=200):
    idx = pd.date_range("2016-01-01", periods=n, freq="h")
    values = 10 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.RandomState(0).normal(0, 0.5, n)
    return pd.Series(values, index=idx)


@pytest.fixture(scope="module")
def toy_fit():
    series = _toy_series(200)
    train = series.iloc[:-48]
    test = series.iloc[-48:]
    # small, fast order for a structural test -- not a real order-selection test
    results = fit_sarimax(train, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), maxiter=50)
    return train, test, results


def test_rolling_forecast_covers_full_test_period(toy_fit):
    train, test, results = toy_fit
    out = rolling_sarimax_forecast(results, test, block_size=24)

    for key in ["forecast", "lower", "upper"]:
        assert len(out[key]) == len(test)
        assert list(out[key].index) == list(test.index)


def test_rolling_forecast_confidence_interval_contains_mean(toy_fit):
    train, test, results = toy_fit
    out = rolling_sarimax_forecast(results, test, block_size=24)

    assert (out["lower"].values <= out["forecast"].values).all()
    assert (out["forecast"].values <= out["upper"].values).all()


def test_rolling_forecast_raises_on_non_multiple_length(toy_fit):
    train, test, results = toy_fit
    with pytest.raises(ValueError):
        rolling_sarimax_forecast(results, test.iloc[:-1], block_size=24)
