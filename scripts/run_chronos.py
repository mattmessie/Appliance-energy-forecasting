"""
scripts/run_chronos.py

Part 7: foundation model (Chronos).

MUST BE RUN LOCALLY, NOT IN THIS DEVELOPMENT SANDBOX -- it downloads
pretrained weights from Hugging Face Hub the first time it runs, and this
sandbox's network policy blocks huggingface.co. On a normal machine with
internet access this should just work: `pip install chronos-forecasting
torch` (already in requirements.txt), then `python scripts/run_chronos.py`.

Uses amazon/chronos-t5-tiny by default (smallest/fastest of the Chronos
family) zero-shot, target-only (see appliance_energy.models.foundation for
why: Chronos's API takes a single numeric context series, with no mechanism
for passing weather/sensor covariates the way SARIMAX/XGBoost can). Same
rolling 14-daily-origin design as the rest of the pipeline.

Saves:
    outputs/figures/foundation_model_forecast.png
    Updated outputs/forecasts/all_forecasts.csv, outputs/metrics/model_comparison.csv
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.config import (
    PROCESSED_DIR, FORECAST_DIR, METRICS_DIR, FIGURE_DIR,
    TARGET, DAILY_PERIOD, TEST_STEPS,
)
from appliance_energy.evaluation import evaluate_all
from appliance_energy.models.foundation import load_chronos_pipeline, rolling_chronos_forecast

MODEL_NAME = "amazon/chronos-t5-tiny"


def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "appliance_hourly.csv", index_col="date", parse_dates=True)
    return df.asfreq("h")


def plot_forecast(train: pd.Series, test: pd.Series, result: dict):
    fig, ax = plt.subplots(figsize=(14, 6))

    context = train.iloc[-DAILY_PERIOD * 21 :]
    context.plot(ax=ax, label="train (context)", color="black", linewidth=0.8)
    test.plot(ax=ax, label="actual (test)", color="black", linewidth=1.4)

    fc = result["forecast"]
    fc.plot(ax=ax, label=f"foundation_model ({MODEL_NAME}, zero-shot)", color="tab:orange", linewidth=1.0)
    ax.fill_between(
        fc.index, result["lower"], result["upper"],
        color="tab:orange", alpha=0.15, label="90% interval",
    )

    for day_start in test.index[::DAILY_PERIOD][1:]:
        ax.axvline(day_start, color="grey", linestyle=":", linewidth=0.4)
    ax.axvline(test.index[0], color="grey", linestyle="--", linewidth=0.8)

    ax.set_title(f"Chronos ({MODEL_NAME}) zero-shot rolling 24h-ahead forecast vs actual")
    ax.set_ylabel("Appliances (Wh)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "foundation_model_forecast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_hourly()
    y = df[TARGET]
    train, test = y.iloc[:-TEST_STEPS], y.iloc[-TEST_STEPS:]

    print(f"Loading Chronos pipeline ({MODEL_NAME})... this downloads pretrained "
          f"weights the first time and may take a minute.")
    t0 = time.time()
    pipeline = load_chronos_pipeline(MODEL_NAME)
    print(f"Loaded in {time.time()-t0:.1f}s")

    print("\nRolling Chronos forward across the 14-day test period (zero-shot, no fitting)...")
    t0 = time.time()
    result = rolling_chronos_forecast(pipeline, train, test, block_size=DAILY_PERIOD)
    print(f"Rolling forecast done in {time.time()-t0:.1f}s")

    plot_forecast(train, test, result)

    # --- Merge into existing outputs ---
    print("\nMerging into existing forecasts/metrics...")
    all_forecasts = pd.read_csv(FORECAST_DIR / "all_forecasts.csv", index_col="date", parse_dates=True)
    all_forecasts["foundation_model"] = result["forecast"]
    all_forecasts["foundation_model_lower"] = result["lower"]
    all_forecasts["foundation_model_upper"] = result["upper"]
    all_forecasts.to_csv(FORECAST_DIR / "all_forecasts.csv")

    model_cols = [
        "mean", "naive", "seasonal_naive_daily", "seasonal_naive_weekly", "drift",
        "sarimax", "sarimax_exog", "feature_model",
    ]
    model_forecasts = {c: all_forecasts[c] for c in model_cols if c in all_forecasts.columns}
    model_forecasts["foundation_model"] = result["forecast"]

    comparison = evaluate_all(test, model_forecasts, train)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print("\n" + comparison.to_string(index=False))

    print(f"\nSaved updated forecasts to {FORECAST_DIR / 'all_forecasts.csv'}")
    print(f"Saved updated comparison to {METRICS_DIR / 'model_comparison.csv'}")
    print(f"Saved forecast plot to {FIGURE_DIR / 'foundation_model_forecast.png'}")
    print(
        "\nNext step: send these three files back (or the whole repo folder) "
        "so the write-up in reports/report.md Section 8 can be finished."
    )


if __name__ == "__main__":
    main()
