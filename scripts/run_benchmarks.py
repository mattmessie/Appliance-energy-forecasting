"""
scripts/run_benchmarks.py

Part 2 (train/test split) + Part 3: benchmark models.

Splits the hourly Appliances series into an initial train set and a 336-hour
(14-day) test period, then runs a ROLLING 24-hour-ahead evaluation: 14 daily
origins, each forecasting the next day using an expanding history (train +
every previously-revealed test day). See report.md, Section 4, for why this
design was chosen over a single long-range forecast.

Saves:
    outputs/forecasts/all_forecasts.csv   (actual + each benchmark forecast)
    outputs/metrics/model_comparison.csv  (metric table, sorted by MASE)
    outputs/figures/forecast_comparison.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.config import (
    PROCESSED_DIR,
    FORECAST_DIR,
    METRICS_DIR,
    FIGURE_DIR,
    TARGET,
    DAILY_PERIOD,
    WEEKLY_PERIOD,
    TEST_STEPS,
)
from appliance_energy.models.benchmarks import generate_rolling_benchmarks
from appliance_energy.evaluation import evaluate_all


def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "appliance_hourly.csv", index_col="date", parse_dates=True)
    return df


def train_test_split(y: pd.Series, test_steps: int = TEST_STEPS):
    train = y.iloc[:-test_steps]
    test = y.iloc[-test_steps:]
    return train, test


def plot_forecast_comparison(train: pd.Series, test: pd.Series, forecasts: dict):
    fig, ax = plt.subplots(figsize=(14, 6))

    # Show the last 3 weeks of training context, then the full test period.
    context = train.iloc[-DAILY_PERIOD * 21 :]
    context.plot(ax=ax, label="train (context)", color="black", linewidth=0.8)
    test.plot(ax=ax, label="actual (test)", color="black", linewidth=1.4)

    for name, fc in forecasts.items():
        fc.plot(ax=ax, label=name, linewidth=1.0, alpha=0.85)

    # Mark each daily rolling-origin boundary so the "re-forecast every day"
    # behaviour is visible (each origin restarts flat/tiled benchmarks).
    ax.axvline(test.index[0], color="grey", linestyle="--", linewidth=0.8)
    for day_start in test.index[::DAILY_PERIOD][1:]:
        ax.axvline(day_start, color="grey", linestyle=":", linewidth=0.4)

    ax.set_title(
        "Benchmark forecasts vs actual — rolling 24h-ahead forecasts, "
        "final 14 days (test period)"
    )
    ax.set_ylabel("Appliances (Wh)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "forecast_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_hourly()
    y = df[TARGET]

    train, test = train_test_split(y, TEST_STEPS)
    print(f"Train: {len(train)} obs ({train.index.min()} to {train.index.max()})")
    print(f"Test:  {len(test)} obs ({test.index.min()} to {test.index.max()})")

    print("\nGenerating rolling 24-hour-ahead benchmark forecasts (14 daily origins)...")
    forecasts = generate_rolling_benchmarks(
        train, test, block_size=DAILY_PERIOD,
        daily_period=DAILY_PERIOD, weekly_period=WEEKLY_PERIOD,
    )

    print("Evaluating forecasts...")
    comparison = evaluate_all(test, forecasts, train)
    print("\n" + comparison.to_string(index=False))

    # Save forecasts (actual + each benchmark), matching the all_forecasts.csv
    # schema in the README (further model columns appended in later parts).
    all_forecasts = pd.DataFrame({"actual": test})
    for name, fc in forecasts.items():
        all_forecasts[name] = fc
    all_forecasts.to_csv(FORECAST_DIR / "all_forecasts.csv")
    print(f"\nSaved forecasts to {FORECAST_DIR / 'all_forecasts.csv'}")

    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print(f"Saved metric comparison to {METRICS_DIR / 'model_comparison.csv'}")

    plot_forecast_comparison(train, test, forecasts)
    print(f"Saved plot to {FIGURE_DIR / 'forecast_comparison.png'}")


if __name__ == "__main__":
    main()
