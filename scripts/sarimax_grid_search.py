"""
scripts/sarimax_grid_search.py

Part 4 (order selection): AIC grid search over p=[0,6], d=[0,2], q=[0,6]
(147 combinations), seasonal_order fixed at (1, 1, 1, 24) per the
assignment's suggested starting point (captures daily seasonality).

For speed, the grid search itself runs on the last 30 days of the training
set (720 hourly observations = 30 full daily cycles -- plenty to estimate a
period-24 seasonal model) rather than the full ~3000-observation training
set. A full 147-combination grid on the full training set was benchmarked
at 25-130s per fit (several hours total); on the 30-day subset it's
5-25s per fit (~30-40 minutes total). The order chosen by AIC on this
subset is then refit on the FULL training set in a separate step
(fit_best_sarimax.py) for the actual rolling forecast -- this is standard
practice for speeding up order selection without compromising the final
fitted model.

Writes incrementally to outputs/metrics/sarimax_grid_search.csv after every
fit (so the run is resumable and can be inspected while still in progress).
"""

import argparse
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.config import PROCESSED_DIR, METRICS_DIR, TARGET, TEST_STEPS

from statsmodels.tsa.statespace.sarimax import SARIMAX

SEASONAL_ORDER = (1, 1, 1, 24)
P_RANGE = range(0, 7)
D_RANGE = range(0, 3)
Q_RANGE = range(0, 7)
GRID_SEARCH_DAYS = 30  # subset of train used for order selection only
MAXITER = 35

OUT_PATH = METRICS_DIR / "sarimax_grid_search.csv"


def load_grid_search_series() -> pd.Series:
    df = pd.read_csv(PROCESSED_DIR / "appliance_hourly.csv", index_col="date", parse_dates=True)
    df = df.asfreq("h")
    train = df[TARGET].iloc[:-TEST_STEPS]
    subset = train.iloc[-GRID_SEARCH_DAYS * 24 :]
    return subset


def already_done() -> set:
    """Read any existing results so a partial run can resume without
    repeating fits."""
    if not OUT_PATH.exists():
        return set()
    existing = pd.read_csv(OUT_PATH)
    return set(zip(existing["p"], existing["d"], existing["q"]))


def append_result(row: dict):
    df_row = pd.DataFrame([row])
    header = not OUT_PATH.exists()
    df_row.to_csv(OUT_PATH, mode="a", header=header, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-fits", type=int, default=None,
        help="Only run up to this many remaining fits this invocation (for "
             "running the grid search in bounded batches; resumable via the "
             "existing results CSV).",
    )
    args = parser.parse_args()

    series = load_grid_search_series()
    print(f"Grid search series: {len(series)} obs ({series.index.min()} to {series.index.max()})")

    combos = list(product(P_RANGE, D_RANGE, Q_RANGE))
    done = already_done()
    remaining = [c for c in combos if c not in done]
    print(f"{len(combos)} total combinations, {len(done)} already done, {len(remaining)} remaining.")

    if args.max_fits is not None:
        remaining = remaining[: args.max_fits]
        print(f"Running this batch: {len(remaining)} fits.")

    for i, (p, d, q) in enumerate(remaining):
        t0 = time.time()
        try:
            model = SARIMAX(
                series,
                order=(p, d, q),
                seasonal_order=SEASONAL_ORDER,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False, maxiter=MAXITER)
            aic = res.aic
            bic = res.bic
            converged = bool(res.mle_retvals.get("converged", False))
            error = ""
        except Exception as exc:  # noqa: BLE001 -- log and continue the grid
            aic = float("inf")
            bic = float("inf")
            converged = False
            error = str(exc)[:200]

        elapsed = time.time() - t0
        append_result(
            {
                "p": p, "d": d, "q": q,
                "seasonal_order": str(SEASONAL_ORDER),
                "aic": aic, "bic": bic,
                "converged": converged,
                "fit_seconds": round(elapsed, 1),
                "error": error,
            }
        )
        print(
            f"[{i+1}/{len(remaining)}] order=({p},{d},{q}) "
            f"aic={aic:.1f} converged={converged} time={elapsed:.1f}s"
        )

    print(f"\nDone. Results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
