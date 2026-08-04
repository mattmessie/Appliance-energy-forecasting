"""
scripts/run_feature_model.py

Part 5 (covariates/features) + Part 6: feature-based ML model (XGBoost).

Feature table: original sensor/weather columns + time features (hour,
dayofweek, is_weekend, cyclic encodings) + target lag/rolling features
(lags 1,2,3,6,12,24,48,168; rolling mean/std over 3,6,12,24,168 hours,
shift(1)'d before rolling -- see appliance_energy.features).

Model: XGBoost, tuned via RandomizedSearchCV with TimeSeriesSplit (fit ONCE
on the initial training set, per the rolling design established in Parts
3-4 -- see report.md, Section 4).

Rolling forecast: 14 daily origins, recursive within each 24h block for
short lags (see appliance_energy.models.feature_models docstring). Sensor/
weather covariates use their REAL test-period values -- this makes the
result a CONDITIONAL forecast, not a true forecast (see Part 9 Q5); this
script says so explicitly rather than treating it as a true forecast.

Saves:
    outputs/figures/feature_importance.png
    outputs/figures/feature_model_forecast.png
    Updated outputs/forecasts/all_forecasts.csv, outputs/metrics/model_comparison.csv
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.config import (
    PROCESSED_DIR, FORECAST_DIR, METRICS_DIR, FIGURE_DIR, MODEL_DIR,
    TARGET, DAILY_PERIOD, TEST_STEPS,
)
from appliance_energy.evaluation import evaluate_all
from appliance_energy.features import make_feature_table, feature_columns, DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS
from appliance_energy.models.feature_models import fit_feature_model, rolling_feature_forecast


def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "appliance_hourly.csv", index_col="date", parse_dates=True)
    return df.asfreq("h")


def plot_feature_importance(model, feature_cols):
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top = importances.head(20)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_title("XGBoost feature importance (top 20)")
    ax.set_xlabel("Importance (gain-based, sklearn default)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return importances


def plot_feature_forecast(train: pd.Series, test: pd.Series, forecast: pd.Series):
    fig, ax = plt.subplots(figsize=(14, 6))

    context = train.iloc[-DAILY_PERIOD * 21 :]
    context.plot(ax=ax, label="train (context)", color="black", linewidth=0.8)
    test.plot(ax=ax, label="actual (test)", color="black", linewidth=1.4)
    forecast.plot(ax=ax, label="feature_model (XGBoost, conditional)", color="tab:green", linewidth=1.0)

    for day_start in test.index[::DAILY_PERIOD][1:]:
        ax.axvline(day_start, color="grey", linestyle=":", linewidth=0.4)
    ax.axvline(test.index[0], color="grey", linestyle="--", linewidth=0.8)

    ax.set_title(
        "Feature-based (XGBoost) rolling 24h-ahead forecast vs actual\n"
        "(conditional on real future weather -- see report.md Part 9 Q5)"
    )
    ax.set_ylabel("Appliances (Wh)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_model_forecast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_hourly()
    y = df[TARGET]
    train_raw = df.iloc[:-TEST_STEPS]
    test_raw = df.iloc[-TEST_STEPS:]
    train, test = train_raw[TARGET], test_raw[TARGET]

    print(f"Building feature table (lags={DEFAULT_LAGS}, windows={DEFAULT_ROLLING_WINDOWS})...")
    feature_table = make_feature_table(train_raw, target=TARGET)
    cols = feature_columns(feature_table, TARGET)
    print(f"Feature table: {feature_table.shape[0]} rows (after dropping lag/rolling warm-up), "
          f"{len(cols)} features")

    print("\nTuning XGBoost (RandomizedSearchCV, TimeSeriesSplit, 25 iterations)...")
    model, best_params = fit_feature_model(
        feature_table[cols], feature_table[TARGET], n_iter=25, cv_splits=3
    )
    print(f"Best params: {best_params}")

    with open(MODEL_DIR / "feature_model_best_params.txt", "w") as f:
        f.write(str(best_params))

    print("\nSaving feature importance plot...")
    importances = plot_feature_importance(model, cols)
    print("Top 10 features:\n" + importances.head(10).to_string())

    print("\nRolling XGBoost forward across the 14-day test period (recursive within-block)...")
    forecast = rolling_feature_forecast(
        model, df, train, test, cols, target=TARGET,
        lags=DEFAULT_LAGS, windows=DEFAULT_ROLLING_WINDOWS, block_size=DAILY_PERIOD,
    )

    plot_feature_forecast(train, test, forecast)

    # --- Merge into existing outputs ---
    print("\nMerging into existing forecasts/metrics...")
    all_forecasts = pd.read_csv(FORECAST_DIR / "all_forecasts.csv", index_col="date", parse_dates=True)
    all_forecasts["feature_model"] = forecast
    all_forecasts.to_csv(FORECAST_DIR / "all_forecasts.csv")

    model_forecasts = {
        name: all_forecasts[name]
        for name in ["mean", "naive", "seasonal_naive_daily", "seasonal_naive_weekly", "drift", "sarimax"]
    }
    model_forecasts["feature_model"] = forecast

    comparison = evaluate_all(test, model_forecasts, train)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print("\n" + comparison.to_string(index=False))

    print(f"\nSaved updated forecasts to {FORECAST_DIR / 'all_forecasts.csv'}")
    print(f"Saved updated comparison to {METRICS_DIR / 'model_comparison.csv'}")
    print(f"Saved feature importance plot to {FIGURE_DIR / 'feature_importance.png'}")
    print(f"Saved forecast plot to {FIGURE_DIR / 'feature_model_forecast.png'}")


if __name__ == "__main__":
    main()
