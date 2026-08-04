# Appliance Energy Forecasting — Report

*Draft in progress. Sections are filled in as each part of the pipeline is completed.*

## 1. Introduction

*(to be written — Part 10)*

## 2. Data and preprocessing

*(to be written — Part 10; see `src/appliance_energy/data.py` and
`outputs/metrics/stationarity_summary.txt` for the underlying work)*

## 3. Exploratory analysis

*(to be written — Part 10; see `outputs/figures/eda_*.png`,
`outputs/figures/acf_pacf_*.png`)*

## 4. Forecasting design

**Target.** `Appliances` — household appliance energy use (Wh), resampled from the
original 10-minute readings to an hourly mean. The hourly series runs from
2016-01-11 17:00 to 2016-05-27 18:00 (3,290 observations, no missing values).

**Horizon.** 24 hours, per the assignment's core task ("forecast appliance energy
use over the next 24 hours"). On hourly data this is `horizon = 24`
(`config.DAILY_PERIOD`).

**Train/test split and evaluation design.** Per the assignment brief (Parts
3, 4, 6, 7: "use last 14 days as test period... forecast the next 24
hours"), this is a **rolling/walk-forward evaluation**, not one long-range
forecast:

- **Test period:** the final 14 days of the series = 336 hourly observations
  (`config.TEST_STEPS = 14 * 24`), i.e. 2016-05-13 19:00 onward, split into
  14 consecutive 24-hour blocks (days 1-14 of the test period).
- **Initial train set:** everything before that, 2,954 observations.
- **Rolling origins:** for each of the 14 daily blocks, the model forecasts
  the next 24 hours using an *expanding history* — the original training set
  plus every test day that has already "happened" (i.e. become known) by
  that point. Day 1 of the test period is forecast from the original
  training set alone; day 2 is forecast using training data + the
  (now-actual, known) day 1; and so on through day 14.

This mirrors how a real smart-home forecasting system would actually
operate: re-forecast the next day each morning using the latest available
data, rather than committing to a single forecast 14 days out and never
updating it. It also means seasonal-naive benchmarks (lag 24, lag 168) are
always using *genuinely available* history at each origin — never a value
from later in the test period than the forecast origin — so there is no
leakage despite the test period effectively "growing" the training data as
it unrolls.

**Model fitting under the rolling design.** For the parametric models
(SARIMAX, the feature-based ML model — Parts 4 and 6), refitting from
scratch at all 14 origins would be expensive (SARIMAX's AIC grid search
alone searches up to 7×3×7 = 147 candidate orders) and isn't what the brief
asks for. The approach taken here: **fit once** on the initial training set
(grid search for SARIMAX; a single trained model for the feature-based
approach), then **update state, not parameters**, at each origin — for
SARIMAX via `SARIMAXResults.append(..., refit=False)` to filter the newly
revealed days through the fitted model before forecasting the next 24 hours;
for the feature-based model, by recomputing lag/rolling features from the
expanding history and predicting with the already-trained model. This keeps
the evaluation realistic (every forecast is a genuine 24-hour-ahead forecast
made from only-the-past) without re-estimating parameters 14 times over,
which would be neither expected nor a good use of the assignment's time
budget. The benchmark models below don't need this distinction since they
have no parameters — "fitting" is just reading off the expanding history.

**Evaluation metrics** (`src/appliance_energy/evaluation.py`), computed on
the full 336-hour test period (all 14 rolling-origin forecasts concatenated)
for every model:

- **MAE** — mean absolute error, in Wh.
- **RMSE** — root mean squared error, in Wh (penalises large errors more).
- **MASE** — mean absolute scaled error. Scaled against the in-sample
  one-step-ahead naive forecast on the *training* set
  (`mean(|y_t − y_{t-1}|)` over train), per Hyndman & Koehler (2006). MASE < 1
  means the model beats a naive one-step persistence forecast on average;
  MASE > 1 means it doesn't. This is the primary metric for "beats the
  benchmark" comparisons since it's scale-free and comparable across models.
- **Bias** — mean signed error (`mean(forecast − actual)`); positive means the
  model systematically over-forecasts, negative means it under-forecasts.

## 5. Benchmark models

*(to be written — Part 10; code in `src/appliance_energy/models/benchmarks.py`,
run via `scripts/run_benchmarks.py`)*

## 6. SARIMAX model

**Order selection.** Following the assignment's suggested starting point
(`seasonal_order=(1,1,1,24)` to capture daily seasonality), the non-seasonal
order `(p,d,q)` was chosen by AIC grid search over the full required range
(p∈[0,6], d∈[0,2], q∈[0,6] — 147 combinations). For speed, the grid search
itself ran on the last 30 days of the training set (720 observations = 30
full daily cycles) rather than the full ~3,000-observation training set —
individual fits at this data length take 5-25s vs. 25-130s+ on the full
series, which made the full 147-combination grid tractable (~35 minutes
total instead of several hours). The order chosen by this search was then
refit on the full training set for the actual forecasting model.

A secondary check mattered here: several of the lowest-AIC candidates from
the fast grid search (using a capped `maxiter=35` for speed) had not
actually converged. Refitting the top 4 candidates with a much higher
iteration budget confirmed all of them do converge given enough iterations,
and **reordered the ranking** — `order=(1,1,6)` converges cleanly and has
the best AIC (7235.3) once given room to converge, edging out
`(0,1,6)` (7241.5) and `(6,0,0)` (7246.2, the best of the models that had
already converged in the fast pass). This is a useful cautionary point for
the report: a fast AIC grid search with a low iteration cap can rank models
in a misleading order if convergence itself is iteration-starved — cheap to
check, easy to get wrong.

**Final model:** `order=(1,1,6)`, `seasonal_order=(1,1,1,24)`, `trend='c'`,
refit on the full 2,954-observation training set (AIC 32,486.5, converged).

**Residual diagnostics** (`outputs/figures/sarimax_residual_diagnostics.png`):
the ACF of residuals shows no significant autocorrelation at any lag up to
48 hours — the model has captured essentially all of the linear
autocorrelation and daily-seasonal structure in the series, residuals are
close to white noise. The residual *distribution*, however, is sharply
peaked and right-skewed (mean ≈ -0.2, std ≈ 63.6 Wh, with a long positive
tail out past 300 Wh) rather than Gaussian — a direct echo of the "bursty
appliances" pattern flagged in Part 1's EDA. SARIMAX assumes Gaussian,
constant-variance errors, so it structurally cannot capture occasional large
spikes; this also explains why the 95% confidence intervals
(`outputs/figures/sarimax_forecast.png`) are wide enough to dip below zero
Wh, which is physically impossible for energy use — a real limitation of
the Gaussian assumption worth naming explicitly rather than a modelling
error to fix.

**Rolling forecast.** Following the same walk-forward design as the
benchmarks (Section 4): the model is fit once, then rolled across the 14
daily test origins via `SARIMAXResults.append(..., refit=False)` — updating
the Kalman filter state with each newly-revealed day without re-estimating
parameters — followed by a fresh `get_forecast(24)` (with 95% CI) at each
origin.

**Result: SARIMAX beats the strongest benchmark.**

| model | MAE | RMSE | MASE | Bias |
|---|---|---|---|---|
| **sarimax** | **38.1** | **65.7** | **0.943** | -5.0 |
| seasonal_naive_weekly | 43.5 | 81.4 | 1.077 | -13.2 |
| seasonal_naive_daily | 48.3 | 85.6 | 1.198 | +1.8 |
| mean | 50.3 | 74.9 | 1.246 | -3.3 |
| naive | 85.6 | 110.4 | 2.121 | +51.0 |
| drift | 85.8 | 110.7 | 2.127 | +51.4 |

SARIMAX is the first model to bring MASE below 1 — a ~12% MAE improvement
over the strongest benchmark (`seasonal_naive_weekly`). Visually
(`sarimax_forecast.png`), it tracks the daily on/off shape substantially
better than any benchmark and picks up several — though not all — of the
larger spike events, consistent with the residual diagnostics above.

## 7. Feature-based model

*(to be written — Part 6/10)*

## 8. Foundation model

*(to be written — Part 7/10)*

## 9. Results and error analysis

*(to be written — Part 8/10)*

## 10. Discussion and limitations

*(to be written — Part 10)*

## 11. Conclusion

*(to be written — Part 10)*
