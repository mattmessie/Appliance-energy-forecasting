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

*(to be written — Part 4/10)*

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
