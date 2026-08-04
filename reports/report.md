# Appliance Energy Forecasting — Report

*Draft in progress. Sections are filled in as each part of the pipeline is completed.*

> **Note on scope (single dataset).** The marking rubric's top criterion
> states work "must be done for both datasets and both types of time series
> models." The assignment brief, however, names and links only one dataset
> throughout all 11 parts — the UCI Appliances Energy Prediction CSV — with
> no second dataset referenced anywhere. "Both types of time series models"
> is unambiguous and covered here (the statistical/parametric family —
> SARIMAX — vs. the ML/data-driven family — feature-based model and
> foundation model, per Parts 4, 6, 7). "Both datasets" is read as leftover
> language from a reused rubric template rather than a live requirement,
> since the brief gives no second dataset to use. This assignment was
> completed entirely on the one dataset the brief provides.

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
(p∈[0,6], d∈[0,2], q∈[0,6] — 147 combinations). The seasonal order
`(P,D,Q,24)` was kept fixed at the brief's suggested `(1,1,1,24)` rather
than also grid-searched — the brief's phrasing here ("p,d,q(P,D<Q)") is
ambiguous, but read alongside the explicit "simple starting point" framing
for `seasonal_order`, the non-seasonal grid search over p,d,q reads as the
intended scope. Grid-searching the seasonal order too would multiply the
search space considerably for a component that mainly takes small values
(0-2) in practice, for uncertain AIC benefit relative to the extra runtime.
For speed, the grid search itself ran on the last 30 days of the training set (720 observations = 30
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

**A genuine tension worth naming, not glossing over:** Part 1's stationarity
tests (ADF and KPSS) both agreed the raw hourly series is already stationary
at levels, which would suggest `d=0` is the "correct" choice. The
AIC-selected model instead has `d=1`. This isn't a contradiction to explain
away — stationarity tests describe the *whole* series' long-run behaviour,
while AIC is optimising one-step-ahead prediction fit on the grid-search
window; a model with mild differencing can still fit the local dynamics
better even when the undifferenced series formally passes stationarity
tests. It's a reminder that "is the series stationary" and "does
differencing improve this particular model's fit" are related but distinct
questions — worth stating explicitly in Section 9 rather than silently
picking whichever answer looks more consistent.

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

### 6b. SARIMAX with exogenous variables

The task overview (item 4) and Part 9 Q2 both ask specifically about
SARIMAX "with exogenous variables where justified" — not just a target-only
model. A second SARIMAX was fit with the same order (`(1,1,6)`,
`seasonal_order=(1,1,1,24)`) plus three exogenous weather covariates:
`T_out`, `RH_out`, `Windspeed`. These three were *selected*, not just
included wholesale from the README's suggested candidate list — screened by
simple correlation with the target: `T_out` (r=0.127), `RH_out` (r=-0.195),
`Windspeed` (r=0.112) all clear the roughly-0.1 threshold; `Visibility`
(r=-0.003), `Tdewpoint` (r=0.021), and `Press_mm_hg` (r=-0.044) are
essentially uncorrelated and were left out. Time-based exogenous features
(hour_sin/cos etc., which the README also suggests) were deliberately not
added here either: `seasonal_order=(1,1,1,24)` already models daily
structure directly, so encoding it a second time as exog risks
multicollinearity for no obvious benefit — this is a difference from the
feature-based model in Section 7, which has no seasonal term of its own and
genuinely needs those time features.

Like the feature-based model below, this uses the *real* test-period values
of the weather covariates — a **conditional forecast** (conditional on
realised future weather), not a true forecast made from only information
available at the origin (see Part 9 Q5).

**Result: exogenous variables help in-sample, not out-of-sample.**

| model | MAE | RMSE | MASE | Bias |
|---|---|---|---|---|
| sarimax (target-only) | 38.1 | 65.7 | 0.943 | -5.0 |
| sarimax_exog | 39.7 | 65.8 | 0.984 | +1.5 |

AIC *improves* with the exogenous variables added (32,351.6 vs. 32,486.5 for
target-only — a genuine in-sample fit improvement), but the rolling
out-of-sample MASE is slightly *worse* (0.984 vs. 0.943). This is a useful,
non-obvious result rather than a modelling mistake to explain away: a
better in-sample likelihood doesn't guarantee better out-of-sample
rolling-forecast accuracy — the weather covariates may be capturing
small amounts of genuine signal that the AIC rewards, while also adding
enough estimation noise/variance across 14 refreshed origins that the net
effect on held-out accuracy is roughly a wash, slightly negative here. Bias
also flips sign (-5.0 to +1.5), suggesting the exogenous version's errors
are less systematically one-directional but not meaningfully smaller.
Visually (`sarimax_exog_comparison.png`) the two forecasts are close to
indistinguishable — this is a small effect in both directions, not a
dramatic failure of either approach.

## 7. Feature-based model

**Feature table** (`src/appliance_energy/features.py`), three sources per
the brief:

- **Original measured variables**: all indoor temperature/humidity
  (T1-T9, RH_1-RH_9) and outdoor weather (T_out, Press_mm_hg, RH_out,
  Windspeed, Visibility, Tdewpoint) columns, used as-is. `lights` (a
  separate appliance circuit that would itself need forecasting, like
  weather — not in the brief's required covariate list) and `rv1`/`rv2`
  (documented in the original UCI dataset release as synthetic
  random-noise columns, included only to test feature-selection
  robustness) are excluded.
- **Time-based features**: hour, day-of-week, weekend indicator, and their
  sine/cosine cyclic encodings — always known in advance, no leakage risk.
- **Lag and rolling features**, built from the target only: lags at
  1, 2, 3, 6, 12, 24, 48, 168 hours; rolling mean and std over 3, 6, 12,
  24, 168-hour windows. Every rolling feature calls `.shift(1)` before
  `.rolling(...)`, so a window ending "at" hour t never includes the value
  at t itself.

**Recursive rolling forecast.** Same 14-daily-origin walk-forward design as
the rest of the pipeline, but with an added wrinkle specific to feature-based
models: within a single 24-hour forecast block, short lags (1, 2, 3, 6, 12)
and their rolling windows reference timestamps *inside that same block* for
every hour after the first — which aren't real yet at the point the block is
being forecast. Per the brief's leakage rule ("lagged and rolling features
use only past observations"), those values are built from the model's own
earlier predictions within the same block, not from the real future: the
forecast proceeds one hour at a time, treating each prediction as if
observed before building the next hour's features. Lags of 24+ hours always
reference a point outside the current block, so they're always built from
genuinely revealed actual data. The model itself (XGBoost) is trained once
and never refit during this process — only the input features change at
each step (`src/appliance_energy/models/feature_models.py`).

**Sensor/weather covariates use their real test-period values** — the same
conditional-forecast choice made for SARIMAX-with-exog above, for the same
reason (forecasting the weather itself is out of scope). See Part 9 Q5.

**Hyperparameter tuning.** `RandomizedSearchCV` (25 iterations) over
`n_estimators`, `max_depth`, `learning_rate`, `subsample`,
`colsample_bytree`, `min_child_weight`, scored on negative MAE, with
`TimeSeriesSplit` (3 folds) rather than a shuffled K-fold — the data is a
time series, so cross-validation folds must respect chronological order
(training only on the past relative to each validation fold) or tuning
itself would leak future information into model selection. Best
parameters: `max_depth=2` (shallow trees — sensible for a feature set this
size relative to ~2,800 training rows), `learning_rate=0.03`,
`n_estimators=100`, `subsample=1.0`, `colsample_bytree=0.6`,
`min_child_weight=1`.

**Feature importance** (`outputs/figures/feature_importance.png`): the
target's own recent history dominates — `roll_mean_3`, `roll_std_3`, and
`lag_1` are the three most important features by a wide margin, followed by
`hour_cos`/`hour`/`lag_2`. The longer lags (`lag_24`, `lag_168`,
`roll_mean_168`) register but rank well below the short-term features.
Sensor/weather variables (`RH_3`, `RH_1`, `RH_7`, `T4`) appear only at the
bottom of the top 20, with importances an order of magnitude smaller than
the leading lag/rolling features. This directly answers part of Part 9 Q3:
recent target history is by far the most useful feature group; time-of-day
is clearly useful too; raw sensor/weather readings contribute comparatively
little once recent-history features are available.

**Result:**

| model | MAE | RMSE | MASE | Bias |
|---|---|---|---|---|
| sarimax | 38.1 | 65.7 | 0.943 | -5.0 |
| **feature_model (XGBoost)** | **42.5** | **66.9** | **1.054** | +0.8 |
| seasonal_naive_weekly | 43.5 | 81.4 | 1.077 | -13.2 |

The feature-based model beats the strongest benchmark (MASE 1.054 vs.
1.077) but does not beat SARIMAX (0.943). Visually
(`feature_model_forecast.png`), it tracks the daily cycle's general shape
but is noticeably *smoother* than the actual series — it systematically
misses the largest spikes and slightly over-predicts the troughs, a
recognisable pattern for tree-based regressors: predictions are bounded by
values seen during training (trees can't extrapolate beyond the leaf
values learned from historical data) and gradient boosting with shallow
trees under this much regularisation tends to regress toward smoother,
averaged behaviour rather than reproducing sharp, rare spikes. The near-zero
bias (+0.8) despite this smoothing suggests the under- and over-prediction
roughly cancel out on average, even though neither the sharp peaks nor
troughs are well captured individually.

## 8. Foundation model

**Model choice: Chronos** (`amazon/chronos-t5-tiny`), used zero-shot —
open-weight and runs entirely locally once downloaded, unlike TimeGPT
(closed, API-only, would require every future run of this repo to have a
valid Nixtla API key just to reproduce results — a poor fit for the brief's
own "runs from a fresh clone" requirement) or TimesFM (a legitimate
alternative, but a historically fussier install for no clear accuracy
edge on data like this).

**Target-only, by necessity, not choice.** Chronos's forecasting API
(`predict_quantiles`) takes a single numeric context series — there's no
mechanism for passing weather/sensor covariates alongside it, unlike
SARIMAX (`exog=`) or the feature-based model (arbitrary feature columns).
This is a genuine capability difference worth naming directly rather than
glossing over: Chronos structurally cannot use the same information the
other two "richer" models can, in Part 9 Q4/Q5's terms.

**Zero-shot means no fitting step at all** — the pretrained model is called
directly with the available history as context; there's no training-set
optimisation the way there is for SARIMAX (AIC grid search) or XGBoost
(RandomizedSearchCV). The same 14-daily-origin rolling design applies —
each origin's context is the expanding history up to that point — but
"rolling" here is simpler than for the other models: since there's no
parameter state, each origin's call to `predict_quantiles` is entirely
independent, not a `.append(refit=False)`-style state update. Point
forecasts and confidence intervals both come directly from
`predict_quantiles`'s sample-quantile output — median for the point
forecast, 5th/95th percentiles for a 90% interval.

**A note on how this section was built, for transparency.** The development
sandbox this project was built in blocks network access to
`huggingface.co`, so the pretrained Chronos weights could not actually be
downloaded or run there (verified directly: a request to `huggingface.co`
returns HTTP 403, `host_not_allowed`). What *was* possible from that
sandbox: installing `chronos-forecasting` itself from PyPI (no Hugging Face
access needed for that) and checking the code in
`src/appliance_energy/models/foundation.py` directly against the real
installed library's source — which caught a real bug before it ever ran
(the predict method's argument is `inputs=`, not `context=` as first
assumed) — plus a full unit-test suite
(`tests/test_foundation.py`) exercising the rolling/indexing/no-leakage
logic against a stub pipeline with the exact same tensor shapes as the real
one. Actual inference against the real pretrained weights was run
separately, locally, via `scripts/run_chronos.py`.

*(Results and evaluation table below to be filled in once
`scripts/run_chronos.py` has been run and its output merged.)*

**Result, run locally (see note above on sandbox/execution split):**

| model | MAE | RMSE | MASE | Bias |
|---|---|---|---|---|
| sarimax | 38.1 | 65.7 | **0.943** | -5.0 |
| **foundation_model (Chronos)** | **38.7** | **78.8** | 0.959 | **-34.2** |
| sarimax_exog | 39.7 | 65.8 | 0.984 | +1.5 |
| feature_model (XGBoost) | 42.5 | 66.9 | 1.054 | +0.8 |
| seasonal_naive_weekly | 43.5 | 81.4 | 1.077 | -13.2 |

**Not a clean win or loss — a genuinely mixed result worth reading
carefully.** On MAE and MASE, Chronos is essentially tied with SARIMAX
(0.959 vs 0.943) — remarkable given it had zero fitting, tuning, or
exposure to this specific dataset at all; every other model here required
either an AIC grid search, hyperparameter tuning, or both. But its RMSE is
markedly worse than SARIMAX's (78.8 vs 65.7 — closer to the weakest
benchmark than to SARIMAX), and its bias (-34.2) is by far the worst of any
model tested — roughly 2.5x the size of the next-worst bias
(`seasonal_naive_weekly`, -13.2). Since RMSE penalises large errors more
than MAE does, and MASE is itself a mean (not a max or a spike-sensitive
measure), this combination — competitive *typical-case* accuracy alongside
much worse RMSE and a large systematic negative bias — points to Chronos
consistently *under*-forecasting, most likely concentrated on the large
spike events that the rest of this dataset's EDA (Part 1) already flagged
as the hardest part of this series to model. A zero-shot model with no
exposure to this specific household's consumption baseline, working from
only a capped 512-hour trailing context window, would plausibly default
toward more conservative/central predictions than a model fit directly on
~3,000 hours of this exact series (SARIMAX) or one with lag/rolling
features built directly from it (XGBoost).

Visually (`foundation_model_forecast.png`), this reads exactly as the
metrics suggest: the daily on/off shape and the low troughs are tracked
reasonably well, but almost every large spike (e.g. 16th, 21st, 27th May)
is substantially under-predicted — the orange line stays close to its
typical daily range while the actual series spikes well above it. Beyond
the point forecast, the 90% interval band itself is often too narrow to
even *contain* these spikes — not just a biased point estimate, but a
miscalibrated uncertainty estimate that understates how far off the model
could plausibly be. For a model with literally no exposure to this
household's data, tracking the routine daily pattern this well is a
genuinely notable result; the systematic failure to anticipate or bound the
large spikes is the clear, specific weakness.

## 9. Results and error analysis

**Full model comparison, ranked by MASE, with explicit comparison against
the strongest benchmark** (`seasonal_naive_weekly`; see
`outputs/metrics/vs_strongest_benchmark.csv`):

| model | MAE | RMSE | MASE | Bias | % MAE vs. strongest benchmark | Beats strongest benchmark? |
|---|---|---|---|---|---|---|
| sarimax | 38.1 | 65.7 | 0.943 | -5.0 | +12.4% | ✅ |
| foundation_model | 38.7 | 78.3 | 0.959 | -33.7 | +11.0% | ✅ |
| sarimax_exog | 39.7 | 65.8 | 0.984 | +1.5 | +8.7% | ✅ |
| feature_model | 42.5 | 66.9 | 1.054 | +0.8 | +2.1% | ✅ |
| seasonal_naive_weekly | 43.5 | 81.4 | 1.077 | -13.2 | — | (itself) |
| seasonal_naive_daily | 48.3 | 85.6 | 1.198 | +1.8 | -11.2% | ❌ |
| mean | 50.3 | 74.9 | 1.246 | -3.3 | -15.6% | ❌ |
| naive | 85.6 | 110.4 | 2.121 | +51.0 | -96.9% | ❌ |
| drift | 85.8 | 110.7 | 2.127 | +51.4 | -97.4% | ❌ |

All four "advanced" models (SARIMAX target-only, Chronos, SARIMAX-exog,
XGBoost) beat the strongest benchmark on MASE; none of the five benchmark
models do better than `seasonal_naive_weekly` itself. The ranking is
consistent whether measured by MASE or by % MAE improvement, which is a
useful sanity check — the two metrics agree on model ordering even though
they're scaled differently.

**Error distribution per model** (`error_diagnostics.png`, left panel):
SARIMAX and the feature-based model have the tightest, most centred error
boxes (narrow interquartile range, median close to zero) — the "best
individual-hour behaviour" view, not just best on-average. `naive` and
`drift` are dramatically worse in both spread and median offset, both
sitting clearly above zero (systematic over-forecasting — a direct visual
confirmation of their large positive Bias values in the table above).
`foundation_model`'s box is visibly shifted *below* zero relative to
everything except `seasonal_naive_weekly`, matching its large negative
Bias — a distinct failure mode from `naive`/`drift`'s over-forecasting.

**MAE by day of the 14-day test period** (`error_diagnostics.png`, right
panel) surfaces something the aggregate metrics alone hide: `naive` and
`drift`'s catastrophic day-1 error (MAE > 240, an order of magnitude worse
than every other model that day) is almost entirely a **single-origin
artefact** — the last training-set hour happened to be an unusually high
spike (~350 Wh), and since these two benchmarks are flat/linear
extrapolations from that one value with no seasonal correction, day 1 pays
the full cost of that one unlucky observation. By day 2 they've already
partially recovered (dropping toward 80-110), because the rolling design
lets them re-anchor on a fresh (more typical) value each day — this is
direct visual evidence for why the rolling walk-forward design (Section 4)
matters: a single-origin, non-rolling evaluation would have let one unlucky
training-set endpoint distort the *entire* 336-hour verdict on these
models, not just one day of it. It's also worth noting `seasonal_naive_weekly`
— the strongest *average* benchmark — has its own bad day (day 7, MAE ≈ 98,
worse than every other model that day): "strongest benchmark" is a
statement about the average across the test period, not a guarantee of
being best on every individual day, and the four advanced models are
visibly more *consistently* good across all 14 days (no single-day spikes
of their own) than any of the five benchmarks.

### 9.1 Answers to the required questions (Part 9)

**Q1. Which benchmark model is strongest — naive, daily seasonal naive,
weekly seasonal naive, or drift — and what does this tell you about the
structure of appliance energy use?**

`seasonal_naive_weekly` (MASE 1.077), ahead of `seasonal_naive_daily`
(1.198), with `naive` and `drift` far behind (2.12+). This ordering is
informative, not just a leaderboard: weekly seasonal naive beating daily
seasonal naive means knowing "the same hour last **week**" predicts better
than "the same hour **yesterday**" — i.e. day-of-week genuinely matters,
consistent with the mild-but-real Fri/Sat bump seen in Part 1's
hour/day-of-week boxplots, on top of the dominant 24-hour on/off cycle.
That `naive` and `drift` — which ignore periodicity entirely — are roughly
twice as bad as either seasonal-naive variant confirms the series is
strongly *periodic* rather than driven mainly by short-term momentum from
the immediately preceding hour.

**Q2. Does SARIMAX improve on the strongest benchmark? Discuss whether
daily seasonality, autocorrelation, and exogenous variables are adequately
captured.**

Yes — both the target-only (MASE 0.943, +12.4% MAE) and exogenous
(0.984, +8.7% MAE) versions beat `seasonal_naive_weekly`. Daily seasonality
and autocorrelation are captured well: the residual ACF (Section 6) shows
no significant autocorrelation at any lag out to 48 hours, meaning the
`(1,1,6)×(1,1,1,24)` structure has absorbed essentially all the linear and
daily-seasonal signal available. What's *not* fully captured is the
residual **distribution** — right-skewed and heavy-tailed rather than
Gaussian, a direct echo of the bursty spike pattern from Part 1's EDA — a
distributional/tail-risk gap, distinct from an autocorrelation-capture
failure. Exogenous variables are only partially "adequately" captured in
the sense that matters operationally: they meaningfully improve in-sample
AIC (32,351.6 vs. 32,486.5) but slightly *worsen* rolling out-of-sample
MASE (Section 6b) — the covariates carry real in-sample signal, but that
signal doesn't reliably transfer into better rolling forecasts here.

**Q3. Does the feature-based model improve when lag, rolling, time,
sensor, and weather features are added? Which feature groups appear most
useful?**

The full combined feature set (all groups together) does beat the
strongest benchmark (MASE 1.054 vs. 1.077, +2.1% MAE) but not SARIMAX
(0.943). Feature importance (Section 7) makes the *usefulness ranking*
unambiguous: recent target history (`roll_mean_3`, `roll_std_3`, `lag_1`)
dominates by a wide margin, time-of-day features (`hour_cos`, `hour`)
matter clearly but less, longer lags (`lag_24`, `lag_168`) register but
rank well below the short-term features, and sensor/weather variables
contribute only marginally — appearing at the bottom of the top 20 with
importances an order of magnitude smaller than the leading lag/rolling
features. *Caveat worth naming honestly:* this is feature-importance
evidence from one model fit on the full combined set, not a formal
group-by-group ablation (train once with only lags, once with only
weather, etc., and compare). The ranking is a fair, standard proxy for
"which groups matter" but a stricter causal claim would need the ablation
— noted as a natural extension in Section 10.

**Q4. Does the foundation model outperform the simpler benchmark, SARIMAX,
and feature-based models? Is the improvement, if any, large enough to
justify the extra complexity?**

A genuinely mixed result (Section 8), not a clean win. Chronos essentially
*ties* SARIMAX on MAE/MASE (0.959 vs. 0.943) despite zero fitting or tuning
— remarkable given every other model here needed either a grid search or
hyperparameter tuning. But it clearly does *not* outperform SARIMAX
overall: RMSE is markedly worse (78.3 vs. 65.7) and bias is by far the
worst of any model tested (-33.7, roughly 2.5x the next-worst), traced
visually to systematic under-prediction of spike events with
under-confident (too-narrow) intervals. It does beat every benchmark and
the feature-based model. On "does the improvement justify the extra
complexity" — there effectively *isn't* a numerical improvement over
SARIMAX to weigh against complexity here: SARIMAX wins on RMSE and bias,
and Chronos's "complexity" is actually inverted from the usual sense — zero
per-dataset tuning effort, but a much heavier fixed dependency footprint
(a full pretrained transformer plus `torch`/`transformers`) than SARIMAX's
lightweight statistical model. For this specific dataset and task, SARIMAX
remains the better choice on the actual numbers.

**Q5. Which covariates would genuinely be known at the forecast origin? If
you use future indoor temperature, humidity, or weather values from the
test set, is this a true forecast or a conditional forecast?**

Time-based features (hour, day-of-week, weekend indicator, cyclic
encodings) are always genuinely known in advance — no issue. Target-derived
lag/rolling features are legitimately known too, *by construction*: lags
≥24h always reference genuinely revealed history, and short lags within a
forecast block are built from the model's own earlier predictions in that
block (Section 7), never from real future values — this is exactly why the
recursive within-block design (and its regression tests) mattered. Indoor
temperature/humidity and outdoor weather variables are a different story:
**this project uses their real, realised test-period values** for both
SARIMAX-with-exog (Section 6b) and the feature-based model (Section 7).
That makes both of those two forecasts **conditional forecasts**
(conditional on realised future weather), not true forecasts made from only
information available at the origin. A real deployment would need either
genuine weather *forecasts* as inputs (importing their own forecast error)
or would need to drop these covariates and accept the resulting accuracy
cost — which, per this project's results, appears to be small either way
(SARIMAX-exog barely differs from target-only; sensor/weather features
rank low in Q3's importance analysis). SARIMAX (target-only) and Chronos
(zero-shot, target-only by construction) are the only two models in this
project that are genuine true forecasts throughout.

**Q6. Based on accuracy, interpretability, uncertainty, computational cost,
and ease of deployment, which model would you recommend for practical
smart-home energy forecasting, and why?**

**SARIMAX (target-only).** Across every dimension asked about:

- *Accuracy:* best MASE (0.943) and best RMSE (65.7) of all eight models
  tested — and unlike the exogenous or feature-based models, achieved as a
  genuine true forecast (Q5), not one propped up by unavailable future
  covariates.
- *Interpretability:* explicit AR/MA/seasonal coefficients and residual
  diagnostics that can be directly inspected and explained — a real
  advantage over an XGBoost ensemble or a black-box pretrained transformer
  when justifying forecasts to a non-technical stakeholder.
- *Uncertainty:* principled 95% confidence intervals from the fitted
  state-space model. These do have a known, documented limitation (the
  Gaussian assumption lets the interval dip below zero Wh, Section 6) —
  but that's an understood property of a well-characterised model, not an
  empirically-discovered miscalibration the way Chronos's too-narrow
  intervals turned out to be (Section 8). The XGBoost model as built here
  provides no native uncertainty estimate at all.
- *Computational cost:* a one-time ~35-minute order-selection cost, but the
  deployed rolling forecast itself is cheap — `append` + `forecast` is
  sub-second per day. Chronos needs no training but carries a much heavier
  inference-time dependency footprint (pretrained transformer, `torch`) for
  an embedded smart-home context; XGBoost needs one fast training run but
  a live sensor/weather feed to stay a true forecast at all (Q5).
- *Ease of deployment:* the `append(refit=False)` rolling pattern maps
  directly onto "re-forecast each morning with the latest reading" — cheap
  incremental updates, no retraining, no external weather dependency.

Chronos is a genuinely compelling *zero-effort baseline* — worth keeping in
mind if the priority is skipping model-specific engineering entirely — but
its worse RMSE and materially miscalibrated intervals make it a weaker
practical choice than SARIMAX for this specific application. The
exogenous-SARIMAX and feature-based models would only become preferable if
genuine forecast weather (not realised) were available and shown to help —
this project's results don't demonstrate that; if anything, Section 6b
shows the opposite (exogenous variables slightly *hurt* rolling accuracy
despite helping in-sample fit).

## 10. Discussion and limitations

*(to be written — Part 10)*

## 11. Conclusion

*(to be written — Part 10)*
