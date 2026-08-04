import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.models.foundation import rolling_chronos_forecast


class StubChronosPipeline:
    """Mimics the real ChronosPipeline.predict_quantiles() interface (same
    input/output shapes, verified against the installed chronos-forecasting
    package's source) without needing the actual pretrained weights or
    internet access. Returns quantiles centred on the last context value
    plus widening noise, which is enough to test the rolling/indexing/
    leakage logic -- NOT a real forecast.
    """

    def __init__(self, noise_std=1.0, seed=0):
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)
        self.calls = []  # record each call's context for inspection in tests

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        self.calls.append(inputs.numpy().copy())
        last_value = float(inputs[-1])
        n_q = len(quantile_levels)
        # shape (batch=1, prediction_length, num_quantiles)
        base = last_value + self.rng.normal(0, self.noise_std, size=(1, prediction_length, 1))
        spread = np.array(quantile_levels).reshape(1, 1, n_q) - 0.5  # centred spread per quantile
        quantiles = base + spread * self.noise_std * 4
        mean = torch.tensor(base[:, :, 0])
        return torch.tensor(quantiles), mean


def _toy_series(n=200):
    idx = pd.date_range("2016-01-01", periods=n, freq="h")
    values = 10 + 5 * np.sin(2 * np.pi * np.arange(n) / 24)
    return pd.Series(values, index=idx)


def test_rolling_forecast_covers_full_test_period():
    series = _toy_series(200)
    train, test = series.iloc[:-48], series.iloc[-48:]
    pipeline = StubChronosPipeline()

    out = rolling_chronos_forecast(pipeline, train, test, block_size=24)

    for key in ["forecast", "lower", "upper"]:
        assert len(out[key]) == len(test)
        assert list(out[key].index) == list(test.index)


def test_confidence_interval_contains_median():
    series = _toy_series(200)
    train, test = series.iloc[:-48], series.iloc[-48:]
    pipeline = StubChronosPipeline()

    out = rolling_chronos_forecast(pipeline, train, test, block_size=24)

    assert (out["lower"].values <= out["forecast"].values).all()
    assert (out["forecast"].values <= out["upper"].values).all()


def test_raises_on_non_multiple_length():
    series = _toy_series(200)
    train, test = series.iloc[:-48], series.iloc[-48:]
    pipeline = StubChronosPipeline()

    with pytest.raises(ValueError):
        rolling_chronos_forecast(pipeline, train, test.iloc[:-1], block_size=24)


def test_context_grows_with_each_revealed_block_no_leakage():
    # Regression guard: at origin k, the context passed to the model must
    # be exactly train + test days already revealed (0..k-1) -- never
    # including the block about to be forecast or anything later.
    series = _toy_series(200)
    train, test = series.iloc[:-72], series.iloc[-72:]  # 3 test days
    pipeline = StubChronosPipeline()

    rolling_chronos_forecast(pipeline, train, test, block_size=24, max_context=1000)

    assert len(pipeline.calls) == 3
    # Origin 0: context length == len(train) exactly (nothing revealed yet).
    assert len(pipeline.calls[0]) == len(train)
    # Origin 1: context length == len(train) + 24 (day 1 revealed).
    assert len(pipeline.calls[1]) == len(train) + 24
    # Origin 2: context length == len(train) + 48 (days 1-2 revealed).
    assert len(pipeline.calls[2]) == len(train) + 48
    # The last value of origin 1's context must be the real actual for the
    # last hour of test day 1, not a placeholder/prediction.
    np.testing.assert_allclose(pipeline.calls[1][-1], test.iloc[23])


def test_max_context_caps_history_length():
    series = _toy_series(300)
    train, test = series.iloc[:-24], series.iloc[-24:]
    pipeline = StubChronosPipeline()

    rolling_chronos_forecast(pipeline, train, test, block_size=24, max_context=50)

    assert len(pipeline.calls[0]) == 50
