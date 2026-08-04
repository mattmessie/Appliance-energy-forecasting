"""
scripts/run_full_comparison.py

Part 8: evaluate ALL models together -- common accuracy metrics (already in
outputs/metrics/model_comparison.csv, one row per model), a single unified
forecast plot, cross-model error diagnostics, and an explicit comparison
against the strongest benchmark.

Requires outputs/forecasts/all_forecasts.csv to already contain every
model's forecast column (mean, naive, seasonal_naive_daily,
seasonal_naive_weekly, drift, sarimax, sarimax_exog, feature_model,
foundation_model) -- i.e. every model-specific script has already been run.

Saves:
    outputs/figures/all_models_forecast_comparison.png
    outputs/figures/error_diagnostics.png
    outputs/metrics/vs_strongest_benchmark.csv
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.config import (
    PROCESSED_DIR, FORECAST_DIR, METRICS_DIR, FIGURE_DIR,
    TARGET, DAILY_PERIOD, TEST_STEPS,
)
from appliance_energy.evaluation import evaluate_all

# All 8 forecasting models (excludes drift/naive from the headline plot's
# main palette only where noted -- all are still in the metrics table).
ALL_MODELS = [
    "mean", "naive", "seasonal_naive_daily", "seasonal_naive_weekly", "drift",
    "sarimax", "sarimax_exog", "feature_model", "foundation_model",
]
STRONGEST_BENCHMARK = "seasonal_naive_weekly"


def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "appliance_hourly.csv", index_col="date", parse_dates=True)
    return df.asfreq("h")


def plot_all_models(train: pd.Series, test: pd.Series, all_forecasts: pd.DataFrame):
    """Single plot: actual vs every model's forecast, over the test period."""
    fig, ax = plt.subplots(figsize=(16, 7))

    context = train.iloc[-DAILY_PERIOD * 10 :]
    context.plot(ax=ax, label="train (context)", color="black", linewidth=0.7, alpha=0.6)
    test.plot(ax=ax, label="actual (test)", color="black", linewidth=1.8)

    colors = {
        "mean": "tab:gray", "naive": "tab:orange", "drift": "tab:purple",
        "seasonal_naive_daily": "tab:green", "seasonal_naive_weekly": "tab:olive",
        "sarimax": "tab:red", "sarimax_exog": "mediumvioletred",
        "feature_model": "tab:cyan", "foundation_model": "tab:brown",
    }
    for model in ALL_MODELS:
        if model not in all_forecasts.columns:
            continue
        all_forecasts[model].plot(ax=ax, label=model, linewidth=0.9, alpha=0.75, color=colors.get(model))

    for day_start in test.index[::DAILY_PERIOD][1:]:
        ax.axvline(day_start, color="grey", linestyle=":", linewidth=0.3)
    ax.axvline(test.index[0], color="grey", linestyle="--", linewidth=0.8)

    ax.set_title("All models: rolling 24h-ahead forecasts vs actual, full 14-day test period")
    ax.set_ylabel("Appliances (Wh)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "all_models_forecast_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_diagnostics(test: pd.Series, all_forecasts: pd.DataFrame):
    """Two-panel error diagnostics across all models:
    (1) boxplot of signed errors per model -- spread and bias at a glance.
    (2) mean absolute error by day of the 14-day test period, per model --
        does accuracy degrade as each day's forecast horizon effectively
        "ages" within a block, or is any model consistently worse on
        particular days (e.g. weekends)?
    """
    models = [m for m in ALL_MODELS if m in all_forecasts.columns]
    errors = pd.DataFrame({m: all_forecasts[m] - test for m in models})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: error distribution per model
    axes[0].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[0].boxplot(
        [errors[m].dropna().values for m in models],
        labels=models, showfliers=True, flierprops={"markersize": 3, "alpha": 0.4},
    )
    axes[0].set_title("Forecast error distribution per model (forecast - actual)")
    axes[0].set_ylabel("Error (Wh)")
    axes[0].tick_params(axis="x", rotation=45)

    # Panel 2: MAE by day of the 14-day test period
    day_of_test = np.repeat(np.arange(1, len(test) // DAILY_PERIOD + 1), DAILY_PERIOD)
    errors["test_day"] = day_of_test[: len(errors)]
    mae_by_day = errors.groupby("test_day")[models].apply(lambda g: g.abs().mean())

    for m in models:
        axes[1].plot(mae_by_day.index, mae_by_day[m], marker="o", markersize=3, linewidth=1, label=m)
    axes[1].set_title("MAE by day of the 14-day rolling test period, per model")
    axes[1].set_xlabel("Test day (1-14)")
    axes[1].set_ylabel("MAE (Wh)")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].set_xticks(range(1, 15))

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "error_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return mae_by_day


def compare_to_strongest_benchmark(comparison: pd.DataFrame) -> pd.DataFrame:
    """Add a column showing each model's % MAE improvement over the
    strongest benchmark (seasonal_naive_weekly)."""
    benchmark_mae = comparison.loc[comparison["model"] == STRONGEST_BENCHMARK, "MAE"].iloc[0]
    out = comparison.copy()
    out["pct_MAE_vs_strongest_benchmark"] = (
        (benchmark_mae - out["MAE"]) / benchmark_mae * 100
    ).round(1)
    out["beats_strongest_benchmark"] = out["MASE"] < comparison.loc[
        comparison["model"] == STRONGEST_BENCHMARK, "MASE"
    ].iloc[0]
    return out


def main():
    df = load_hourly()
    y = df[TARGET]
    train, test = y.iloc[:-TEST_STEPS], y.iloc[-TEST_STEPS:]

    all_forecasts = pd.read_csv(FORECAST_DIR / "all_forecasts.csv", index_col="date", parse_dates=True)
    missing = [m for m in ALL_MODELS if m not in all_forecasts.columns]
    if missing:
        raise ValueError(
            f"all_forecasts.csv is missing columns for: {missing}. "
            f"Run every model-specific script first (see report.md, Section 4-8)."
        )

    print("Generating unified forecast comparison plot...")
    plot_all_models(train, test, all_forecasts)

    print("Generating cross-model error diagnostics...")
    mae_by_day = plot_error_diagnostics(test, all_forecasts)
    print("\nMAE by test day (first/last 3 days shown):")
    print(mae_by_day.iloc[[0, 1, 2, -3, -2, -1]].round(1).to_string())

    print("\nComparing all models against strongest benchmark "
          f"({STRONGEST_BENCHMARK})...")
    comparison = evaluate_all(
        test, {m: all_forecasts[m] for m in ALL_MODELS}, train
    )
    comparison_vs_benchmark = compare_to_strongest_benchmark(comparison)
    comparison_vs_benchmark.to_csv(METRICS_DIR / "vs_strongest_benchmark.csv", index=False)
    print("\n" + comparison_vs_benchmark.to_string(index=False))

    print(f"\nSaved unified plot to {FIGURE_DIR / 'all_models_forecast_comparison.png'}")
    print(f"Saved error diagnostics to {FIGURE_DIR / 'error_diagnostics.png'}")
    print(f"Saved benchmark comparison to {METRICS_DIR / 'vs_strongest_benchmark.csv'}")


if __name__ == "__main__":
    main()
