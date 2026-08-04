"""
scripts/run_sarimax_exog.py

Closes a gap in Part 4's original scope: the task overview explicitly asks
for SARIMAX "using appropriate daily seasonality and selected exogenous
variables where justified," and Part 9 Q2 asks whether exogenous variables
are adequately captured -- neither is answerable from the target-only
model in run_sarimax.py alone.

Exogenous variable selection: screened by correlation with the target
(see report.md, Section 6b). T_out, RH_out, and Windspeed have |corr| > 0.1
with Appliances; Visibility, Tdewpoint, and Press_mm_hg are ~uncorrelated
(<0.05) and excluded. Time-based exogenous features (hour_sin/cos etc.,
suggested in the README) are deliberately NOT added here: SARIMAX's
seasonal_order=(1,1,1,24) already models daily structure directly, so
adding redundant time encodings as exog would risk multicollinearity for
no clear benefit (unlike the feature-based model in Part 6, which has no
seasonal_order term and needs them).

Same order=(1,1,6), seasonal_order=(1,1,1,24) as the target-only model, same
rolling walk-forward design. Exogenous covariates use their REAL test-period
values -- this is a CONDITIONAL forecast (conditional on realised future
weather), consistent with how the feature-based model was handled. See
Part 9 Q5.
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
from appliance_energy.models.sarimax import fit_sarimax, rolling_sarimax_forecast

ORDER = (1, 1, 6)
SEASONAL_ORDER = (1, 1, 1, 24)
EXOG_COLS = ["T_out", "RH_out", "Windspeed"]


def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "appliance_hourly.csv", index_col="date", parse_dates=True)
    return df.asfreq("h")


def main():
    df = load_hourly()
    y = df[TARGET]
    train, test = y.iloc[:-TEST_STEPS], y.iloc[-TEST_STEPS:]
    exog = df[EXOG_COLS]
    exog_train, exog_test = exog.iloc[:-TEST_STEPS], exog.iloc[-TEST_STEPS:]

    print(f"Fitting SARIMAX-with-exog order={ORDER}, seasonal_order={SEASONAL_ORDER}, "
          f"exog={EXOG_COLS} on full train ({len(train)} obs)...")
    results = fit_sarimax(train, order=ORDER, seasonal_order=SEASONAL_ORDER, exog=exog_train)
    print(f"Converged: {results.mle_retvals.get('converged')}, AIC: {results.aic:.1f}")

    summary_path = MODEL_DIR / "sarimax_exog_summary.txt"
    summary_path.write_text(
        f"order: {ORDER}\nseasonal_order: {SEASONAL_ORDER}\nexog: {EXOG_COLS}\n"
        f"trend: c\naic: {results.aic:.2f}\nbic: {results.bic:.2f}\n"
        f"converged: {results.mle_retvals.get('converged')}\n\n"
        f"params:\n{results.params.to_string()}\n"
    )
    print(f"Saved model summary to {summary_path}")

    print("\nRolling SARIMAX-with-exog forward across the 14-day test period...")
    result = rolling_sarimax_forecast(
        results, test, block_size=DAILY_PERIOD, exog_test=exog_test
    )
    forecast = result["forecast"].rename("sarimax_exog")

    # --- Plot: target-only vs with-exog, both against actual ---
    all_forecasts = pd.read_csv(FORECAST_DIR / "all_forecasts.csv", index_col="date", parse_dates=True)
    all_forecasts["sarimax_exog"] = forecast
    all_forecasts.to_csv(FORECAST_DIR / "all_forecasts.csv")

    fig, ax = plt.subplots(figsize=(14, 6))
    context = train.iloc[-DAILY_PERIOD * 14 :]
    context.plot(ax=ax, label="train (context)", color="black", linewidth=0.8)
    test.plot(ax=ax, label="actual (test)", color="black", linewidth=1.4)
    all_forecasts["sarimax"].plot(ax=ax, label="sarimax (target-only)", color="tab:red", linewidth=1.0, alpha=0.8)
    forecast.plot(ax=ax, label="sarimax_exog (T_out, RH_out, Windspeed; conditional)", color="tab:purple", linewidth=1.0)
    for day_start in test.index[::DAILY_PERIOD][1:]:
        ax.axvline(day_start, color="grey", linestyle=":", linewidth=0.4)
    ax.axvline(test.index[0], color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("SARIMAX: target-only vs with exogenous weather variables")
    ax.set_ylabel("Appliances (Wh)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sarimax_exog_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Evaluation: target-only vs with-exog vs strongest benchmark ---
    comparison = evaluate_all(
        test,
        {
            "sarimax": all_forecasts["sarimax"],
            "sarimax_exog": forecast,
            "seasonal_naive_weekly": all_forecasts["seasonal_naive_weekly"],
        },
        train,
    )
    print("\n" + comparison.to_string(index=False))
    comparison.to_csv(METRICS_DIR / "sarimax_exog_comparison.csv", index=False)

    print(f"\nSaved updated forecasts to {FORECAST_DIR / 'all_forecasts.csv'}")
    print(f"Saved comparison to {METRICS_DIR / 'sarimax_exog_comparison.csv'}")
    print(f"Saved plot to {FIGURE_DIR / 'sarimax_exog_comparison.png'}")


if __name__ == "__main__":
    main()
