"""
scripts/eda_and_stationarity.py

Part 1 (remaining): exploratory analysis and stationarity testing on the
hourly Appliances series.

Outputs:
    outputs/figures/eda_full_series.png
    outputs/figures/eda_seasonal_zoom.png
    outputs/figures/eda_hour_dow_boxplots.png
    outputs/figures/eda_seasonal_decompose.png
    outputs/figures/acf_pacf_levels.png
    outputs/figures/acf_pacf_diff1.png
    outputs/metrics/stationarity_summary.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.config import PROCESSED_DIR, FIGURE_DIR, METRICS_DIR, TARGET, DAILY_PERIOD

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def load_hourly():
    df = pd.read_csv(PROCESSED_DIR / "appliance_hourly.csv", index_col="date", parse_dates=True)
    return df


def plot_full_series(df):
    fig, ax = plt.subplots(figsize=(14, 5))
    df[TARGET].plot(ax=ax, linewidth=0.8)
    ax.set_title("Hourly appliance energy use, full period")
    ax.set_ylabel("Appliances (Wh)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_full_series.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_seasonal_zoom(df):
    zoom = df[TARGET].loc["2016-02-01":"2016-02-14"]
    fig, ax = plt.subplots(figsize=(14, 5))
    zoom.plot(ax=ax, linewidth=1.2, marker="o", markersize=2)
    ax.set_title("Two-week zoom (1-14 Feb 2016): daily cycle visible")
    ax.set_ylabel("Appliances (Wh)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_seasonal_zoom.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_hour_dow_boxplots(df):
    tmp = df[[TARGET]].copy()
    tmp["hour"] = tmp.index.hour
    tmp["dayofweek"] = tmp.index.dayofweek

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    tmp.boxplot(column=TARGET, by="hour", ax=axes[0])
    axes[0].set_title("Appliance use by hour of day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Appliances (Wh)")

    tmp.boxplot(column=TARGET, by="dayofweek", ax=axes[1])
    axes[1].set_title("Appliance use by day of week (0=Mon)")
    axes[1].set_xlabel("Day of week")
    axes[1].set_ylabel("Appliances (Wh)")

    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_hour_dow_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_seasonal_decompose(df):
    result = seasonal_decompose(df[TARGET], model="additive", period=DAILY_PERIOD)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_seasonal_decompose.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return result


def plot_acf_pacf(series, filename, title, lags=72):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF - {title}")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF - {title}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_stationarity_tests(series, label):
    lines = [f"--- {label} ---"]

    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(series.dropna(), autolag="AIC")
    lines.append(f"ADF statistic: {adf_stat:.4f}")
    lines.append(f"ADF p-value:   {adf_p:.6f}")
    lines.append(f"ADF lags used: {adf_lags}, nobs: {adf_nobs}")
    lines.append(f"ADF critical values: {adf_crit}")
    lines.append(f"ADF conclusion: {'stationary (reject H0)' if adf_p < 0.05 else 'non-stationary (fail to reject H0)'}")
    lines.append("")

    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series.dropna(), regression="c", nlags="auto")
    lines.append(f"KPSS statistic: {kpss_stat:.4f}")
    lines.append(f"KPSS p-value:   {kpss_p:.6f}")
    lines.append(f"KPSS lags used: {kpss_lags}")
    lines.append(f"KPSS critical values: {kpss_crit}")
    lines.append(f"KPSS conclusion: {'stationary (fail to reject H0)' if kpss_p > 0.05 else 'non-stationary (reject H0)'}")
    lines.append("")

    return "\n".join(lines)


def main():
    df = load_hourly()
    y = df[TARGET]

    print("Generating EDA plots...")
    plot_full_series(df)
    plot_seasonal_zoom(df)
    plot_hour_dow_boxplots(df)
    decomp = plot_seasonal_decompose(df)

    print("Running stationarity tests on levels...")
    plot_acf_pacf(y, "acf_pacf_levels.png", "levels (Appliances)")
    summary_levels = run_stationarity_tests(y, "Levels (Appliances, hourly)")

    print("Running stationarity tests on first difference...")
    y_diff1 = y.diff().dropna()
    plot_acf_pacf(y_diff1, "acf_pacf_diff1.png", "first difference")
    summary_diff1 = run_stationarity_tests(y_diff1, "First difference (d=1)")

    print("Running stationarity tests on seasonal (24h) difference...")
    y_seasonal_diff = y.diff(DAILY_PERIOD).dropna()
    summary_seasonal = run_stationarity_tests(y_seasonal_diff, "Seasonal difference (lag 24)")

    # Basic descriptive stats + component summary
    desc = y.describe()
    seasonal_strength = 1 - (np.var(decomp.resid.dropna()) / np.var((decomp.seasonal + decomp.resid).dropna()))
    trend_strength = 1 - (np.var(decomp.resid.dropna()) / np.var((decomp.trend + decomp.resid).dropna()))

    header = (
        "APPLIANCE ENERGY - EDA & STATIONARITY SUMMARY\n"
        "==============================================\n\n"
        f"Series length: {len(y)} hourly observations "
        f"({y.index.min()} to {y.index.max()})\n\n"
        "Descriptive statistics (Wh):\n"
        f"{desc.to_string()}\n\n"
        f"Approx. seasonal strength (24h, additive decomposition): {seasonal_strength:.3f}\n"
        f"Approx. trend strength: {trend_strength:.3f}\n"
        "(Strength close to 1 = component explains most of the variance "
        "left after removing the other components; Wang, Smith & Hyndman 2006 measure.)\n\n"
    )

    full_text = header + summary_levels + summary_diff1 + summary_seasonal

    out_path = METRICS_DIR / "stationarity_summary.txt"
    out_path.write_text(full_text)

    print(full_text)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
