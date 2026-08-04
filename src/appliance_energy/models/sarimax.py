"""SARIMAX model: fitting and rolling 24-hour-ahead forecasting.

The order (p, d, q) is chosen by AIC grid search (see
scripts/sarimax_grid_search.py and reports/report.md, Section 6) and fixed
here; this module handles fitting it on the training set and rolling it
forward across the test period.

Rolling design (matches src/appliance_energy/models/benchmarks.py): the
model is fit ONCE on the initial training set. At each of the 14 daily
origins in the test period, the newly-revealed day of actuals is filtered
through the model via `SARIMAXResults.append(..., refit=False)` -- which
updates the Kalman filter state using the already-estimated parameters,
without re-running the (expensive) MLE optimisation -- and then the next 24
hours are forecast with `get_forecast()`, which also gives confidence
intervals.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


def fit_sarimax(
    train: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    trend: str = "c",
    maxiter: int = 300,
) -> SARIMAXResults:
    """Fit a SARIMAX model on the training series."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False, maxiter=maxiter)
    return results


def rolling_sarimax_forecast(
    fitted_results: SARIMAXResults,
    test: pd.Series,
    block_size: int = 24,
    alpha: float = 0.05,
) -> dict:
    """Roll a fitted SARIMAX model forward across the test period.

    At each daily origin, forecasts the next `block_size` hours from the
    current filtered state, then updates that state with the newly-revealed
    actuals (refit=False -- parameters are NOT re-estimated) before moving
    to the next origin.

    Parameters
    ----------
    fitted_results : SARIMAXResults
        Model already fit on the initial training set (see `fit_sarimax`).
    test : pd.Series
        Full test period (length must be a multiple of `block_size`).
    block_size : int
        Forecast horizon per origin, in hours.
    alpha : float
        Confidence level for the forecast intervals (0.05 -> 95% CI).

    Returns
    -------
    dict with:
        "forecast": pd.Series of point forecasts across the whole test period
        "lower": pd.Series of lower confidence bound
        "upper": pd.Series of upper confidence bound
    """
    if len(test) % block_size != 0:
        raise ValueError(
            f"Test period length ({len(test)}) is not a multiple of "
            f"block_size ({block_size})."
        )

    n_blocks = len(test) // block_size
    current_results = fitted_results

    forecast_parts, lower_parts, upper_parts = [], [], []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(n_blocks):
            block_index = test.index[k * block_size : (k + 1) * block_size]

            fc = current_results.get_forecast(steps=block_size)
            mean = fc.predicted_mean
            mean.index = block_index
            ci = fc.conf_int(alpha=alpha)
            ci.index = block_index

            forecast_parts.append(mean)
            lower_parts.append(ci.iloc[:, 0])
            upper_parts.append(ci.iloc[:, 1])

            # Reveal this block's actuals and update filter state only
            # (refit=False -- parameters stay fixed at their initial MLE).
            actual_block = test.iloc[k * block_size : (k + 1) * block_size]
            current_results = current_results.append(actual_block, refit=False)

    return {
        "forecast": pd.concat(forecast_parts).rename("sarimax"),
        "lower": pd.concat(lower_parts).rename("sarimax_lower"),
        "upper": pd.concat(upper_parts).rename("sarimax_upper"),
    }
