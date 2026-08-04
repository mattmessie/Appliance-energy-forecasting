"""Forecast evaluation metrics and model comparison utilities.

All metrics are computed on a common test set so that benchmark, SARIMAX,
feature-based, and foundation-model forecasts can be compared directly.

MASE follows Hyndman & Koehler (2006): errors are scaled by the in-sample
mean absolute error of a naive one-step-ahead forecast on the *training*
set. A MASE below 1 means the model beats naive persistence on average.
"""

import numpy as np
import pandas as pd


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(actual.values - forecast.values)))


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((actual.values - forecast.values) ** 2)))


def bias(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean signed error (forecast - actual). Positive = over-forecasts."""
    return float(np.mean(forecast.values - actual.values))


def naive_in_sample_scale(train: pd.Series) -> float:
    """In-sample scaling factor for MASE: MAE of a one-step naive forecast
    on the training set, i.e. mean(|y_t - y_{t-1}|) over train.
    """
    diffs = np.abs(np.diff(train.values))
    scale = float(np.mean(diffs))
    if scale == 0:
        raise ValueError(
            "In-sample naive scale is zero (training series is constant); "
            "MASE is undefined."
        )
    return scale


def mase(actual: pd.Series, forecast: pd.Series, train: pd.Series) -> float:
    """Mean absolute scaled error, scaled by the training set's one-step
    naive in-sample MAE.
    """
    scale = naive_in_sample_scale(train)
    return mae(actual, forecast) / scale


def evaluate_forecast(
    actual: pd.Series, forecast: pd.Series, train: pd.Series
) -> dict:
    """Compute MAE, RMSE, MASE, and Bias for a single forecast."""
    return {
        "MAE": mae(actual, forecast),
        "RMSE": rmse(actual, forecast),
        "MASE": mase(actual, forecast, train),
        "Bias": bias(actual, forecast),
    }


def evaluate_all(
    actual: pd.Series, forecasts: dict, train: pd.Series
) -> pd.DataFrame:
    """Evaluate several forecasts against the same actuals/train set.

    Parameters
    ----------
    actual : pd.Series
        True values over the test period.
    forecasts : dict[str, pd.Series]
        Mapping of model name -> forecast series (same index as `actual`).
    train : pd.Series
        Training series, used for the MASE scaling factor.

    Returns
    -------
    pd.DataFrame with one row per model and columns MAE, RMSE, MASE, Bias,
    sorted by MASE ascending (best model first).
    """
    rows = {}
    for name, fc in forecasts.items():
        fc_aligned = fc.reindex(actual.index)
        if fc_aligned.isna().any():
            raise ValueError(
                f"Forecast '{name}' has missing values over the test index "
                f"after alignment; check length/index match."
            )
        rows[name] = evaluate_forecast(actual, fc_aligned, train)

    result = pd.DataFrame(rows).T
    result.index.name = "model"
    result = result.sort_values("MASE")
    return result.reset_index()
