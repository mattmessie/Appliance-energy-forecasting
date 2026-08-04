"""
scripts/run_sarimax.py

Part 4: SARIMAX model.

Order selection (see scripts/sarimax_grid_search.py): AIC grid search over
p=[0,6], d=[0,2], q=[0,6] with seasonal_order fixed at (1,1,1,24), run on
the last 30 days of training data for speed. Winner: order=(1,1,6) at
AIC=7235.3 (converged; re-verified with a high maxiter after the fast grid
search flagged it as one of several close, non-converged top candidates).

This script:
  1. Refits order=(1,1,6), seasonal_order=(1,1,1,24) on the FULL training
     set (2,954 obs).
  2. Saves residual diagnostics: ACF plot + distribution of residuals.
  3. Rolls the fitted model forward across the 14-day test period (24h
     forecasts, append(refit=False) between origins -- see
     appliance_energy.models.sarimax), with 95% confidence intervals.
  4. Merges the SARIMAX forecast into the existing benchmark comparison
     (outputs/forecasts/all_forecasts.csv, outputs/metrics/model_comparison.csv)
     and regenerates the comparison plot.
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.config import (
    PROCESSED_DIR, FORECAST_DIR, METRICS_DIR, FIGURE_DIR, MODEL_DIR,
    TARGET, DAILY_PERIOD, TEST_STEPS,
)
from appliance_energy.evaluation import evaluate_all
from appliance_energy.models.sarimax import fit_sarimax, rolling_sarimax_forecast

ORDER = (1, 1, 6)
SEASONAL_ORDER = (1, 1, 1, 24)


def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "appliance_hourly.csv", index_col="date", parse_dates=True)
    return df.asfreq("h")


def plot_residual_diagnostics(results):
    resid = results.resid.iloc[SEASONAL_ORDER[3] * 2 :]  # drop initial burn-in

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(resid, lags=48, ax=axes[0])
    axes[0].set_title("ACF of SARIMAX residuals (order=(1,1,6), seasonal=(1,1,1,24))")

    axes[1].hist(resid, bins=50, edgecolor="black", linewidth=0.3)
    axes[1].set_title(f"Distribution of residuals (mean={resid.mean():.2f}, std={resid.std():.2f})")
    axes[1].set_xlabel("Residual (Wh)")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sarimax_residual_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sarimax_forecast(train: pd.Series, test: pd.Series, sarimax_result: dict):
    fig, ax = plt.subplots(figsize=(14, 6))

    context = train.iloc[-DAILY_PERIOD * 21 :]
    context.plot(ax=ax, label="train (context)", color="black", linewidth=0.8)
    test.plot(ax=ax, label="actual (test)", color="black", linewidth=1.4)

    fc = sarimax_result["forecast"]
    lower = sarimax_result["lower"]
    upper = sarimax_result["upper"]

    fc.plot(ax=ax, label="sarimax", color="tab:red", linewidth=1.0)
    ax.fill_between(fc.index, lower, upper, color="tab:red", alpha=0.15, label="sarimax 95% CI")

    for day_start in test.index[::DAILY_PERIOD][1:]:
        ax.axvline(day_start, color="grey", linestyle=":", linewidth=0.4)
    ax.axvline(test.index[0], color="grey", linestyle="--", linewidth=0.8)

    ax.set_title("SARIMAX rolling 24h-ahead forecast vs actual (with 95% CI)")
    ax.set_ylabel("Appliances (Wh)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sarimax_forecast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_hourly()
    y = df[TARGET]
    train = y.iloc[:-TEST_STEPS]
    test = y.iloc[-TEST_STEPS:]

    print(f"Fitting SARIMAX order={ORDER}, seasonal_order={SEASONAL_ORDER} on full train "
          f"({len(train)} obs)...")
    results = fit_sarimax(train, order=ORDER, seasonal_order=SEASONAL_ORDER)
    print(f"Converged: {results.mle_retvals.get('converged')}, AIC: {results.aic:.1f}")

    # Note: the full SARIMAXResults object (with seasonal terms, this model's
    # state dimension works out to ~56) pickles to over a gigabyte, because it
    # retains per-timestep Kalman filter/smoother matrices for the whole
    # training set. That's not worth keeping on disk for a model this cheap
    # to refit, so we save just the essentials (order, params, AIC/BIC) as a
    # small summary instead.
    summary_path = MODEL_DIR / "sarimax_summary.txt"
    summary_path.write_text(
        f"order: {ORDER}\n"
        f"seasonal_order: {SEASONAL_ORDER}\n"
        f"trend: c\n"
        f"aic: {results.aic:.2f}\n"
        f"bic: {results.bic:.2f}\n"
        f"converged: {results.mle_retvals.get('converged')}\n\n"
        f"params:\n{results.params.to_string()}\n"
    )
    print(f"Saved model summary to {summary_path}")

    print("\nSaving residual diagnostics...")
    plot_residual_diagnostics(results)

    print("Rolling SARIMAX forward across the 14-day test period...")
    sarimax_result = rolling_sarimax_forecast(results, test, block_size=DAILY_PERIOD)

    plot_sarimax_forecast(train, test, sarimax_result)

    # --- Merge into existing benchmark outputs ---
    print("\nMerging into existing forecasts/metrics...")
    all_forecasts = pd.read_csv(FORECAST_DIR / "all_forecasts.csv", index_col="date", parse_dates=True)
    all_forecasts["sarimax"] = sarimax_result["forecast"]
    all_forecasts["sarimax_lower"] = sarimax_result["lower"]
    all_forecasts["sarimax_upper"] = sarimax_result["upper"]
    all_forecasts.to_csv(FORECAST_DIR / "all_forecasts.csv")

    benchmark_forecasts = {
        name: all_forecasts[name]
        for name in ["mean", "naive", "seasonal_naive_daily", "seasonal_naive_weekly", "drift"]
    }
    benchmark_forecasts["sarimax"] = sarimax_result["forecast"]

    comparison = evaluate_all(test, benchmark_forecasts, train)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print("\n" + comparison.to_string(index=False))

    print(f"\nSaved updated forecasts to {FORECAST_DIR / 'all_forecasts.csv'}")
    print(f"Saved updated comparison to {METRICS_DIR / 'model_comparison.csv'}")
    print(f"Saved SARIMAX forecast plot to {FIGURE_DIR / 'sarimax_forecast.png'}")
    print(f"Saved residual diagnostics to {FIGURE_DIR / 'sarimax_residual_diagnostics.png'}")


if __name__ == "__main__":
    main()
