# Appliance Energy Forecasting

A time-series case study forecasting household appliance energy use (UCI
"Appliances Energy Prediction" dataset) using five simple benchmarks,
SARIMAX (target-only and with exogenous weather covariates), a tuned
feature-based ML model (XGBoost), and a pretrained time-series foundation
model (Chronos) — evaluated under an identical rolling 24-hour-ahead
walk-forward design across the final 14 days of the series.

**Full write-up:** [`reports/report.docx`](reports/report.docx) /
[`reports/report.pdf`](reports/report.pdf) (the submitted report — 7 pages).
[`reports/report.md`](reports/report.md) is a longer working document with
the full design reasoning, process notes, and every intermediate finding.

## Headline result

| model | MAE | RMSE | MASE | Bias |
|---|---|---|---|---|
| **SARIMAX (target-only)** | **38.1** | **65.7** | **0.943** | -5.0 |
| Foundation model (Chronos, zero-shot) | 38.7 | 78.3 | 0.959 | -33.7 |
| SARIMAX + exogenous weather | 39.7 | 65.8 | 0.984 | +1.5 |
| Feature-based model (XGBoost, tuned) | 42.5 | 66.9 | 1.054 | +0.8 |
| Seasonal naive (weekly) — strongest benchmark | 43.5 | 81.4 | 1.077 | -13.2 |

SARIMAX (target-only) is the best-performing and recommended model — see
`reports/report.docx` Section 9 and the required-questions section for the
full analysis and reasoning.

## Repository structure

```text
appliance-energy-forecasting/
│
├── data/
│   ├── raw/                    # cached raw UCI CSV (gitignored — regenerated on first run)
│   └── processed/              # cleaned, resampled hourly series
│
├── notebooks/                  # exploration & results, one per project phase
│
├── src/appliance_energy/
│   ├── config.py                # paths, constants (target, horizon, test period)
│   ├── data.py                  # download/clean/resample
│   ├── features.py               # time + lag/rolling feature engineering
│   ├── evaluation.py            # MAE, RMSE, MASE, Bias
│   └── models/
│       ├── benchmarks.py         # mean, naive, seasonal naive, drift (rolling)
│       ├── sarimax.py            # SARIMAX fit + rolling forecast (target/exog)
│       ├── feature_models.py     # XGBoost fit (tuned) + recursive rolling forecast
│       └── foundation.py         # Chronos zero-shot rolling forecast
│
├── scripts/
│   ├── run_pipeline.py           # star: single entry point — runs everything
│   ├── sarimax_grid_search.py    # AIC grid search (147 combos, ~35 min)
│   ├── eda_and_stationarity.py
│   ├── run_benchmarks.py
│   ├── run_sarimax.py
│   ├── run_sarimax_exog.py
│   ├── run_feature_model.py
│   ├── run_chronos.py            # needs internet access to huggingface.co
│   └── run_full_comparison.py
│
├── outputs/
│   ├── figures/                  # every plot referenced in the report
│   ├── forecasts/all_forecasts.csv       # every model's forecast, one file
│   └── metrics/model_comparison.csv      # final MAE/RMSE/MASE/Bias table
│
├── reports/
│   ├── report.docx, report.pdf   # the submitted report
│   └── report.md                 # working document with full design reasoning
│
└── tests/                        # pytest — 37 tests, incl. leakage-regression guards
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running the pipeline

```bash
python scripts/run_pipeline.py
```

This downloads/caches the raw data, cleans and resamples it, runs the EDA
and stationarity tests, fits every benchmark and model, evaluates them all
under the rolling walk-forward design, and saves every figure/metric/
forecast to `outputs/`.

Two steps have genuine practical constraints on a fresh clone, handled
explicitly rather than silently:

- **SARIMAX order selection** (the 147-combination AIC grid search) takes
  ~35 minutes. By default the pipeline skips re-running it and uses the
  order already selected (`(1,1,6)`, seasonal `(1,1,1,24)` — see
  `reports/report.docx`, Section 6). Pass `--grid-search` to redo it from
  scratch.
- **The foundation model** (Chronos) downloads pretrained weights from
  Hugging Face Hub on first use, which needs outbound internet access to
  `huggingface.co`. On a normal machine this just works; on a
  network-restricted machine it will fail with a clear message rather than
  crashing the rest of the pipeline. Pass `--skip-chronos` to skip it
  outright.

```bash
python scripts/run_pipeline.py --grid-search      # also redo SARIMAX order selection
python scripts/run_pipeline.py --skip-chronos     # skip the foundation model step
```

Individual steps can also be run on their own — see `scripts/`, each is
self-contained and documented.

## Running the tests

```bash
pytest
```

37 tests across data preprocessing, feature engineering, every model's
rolling-forecast logic, and the evaluation metrics — including regression
tests that specifically guard against data leakage (e.g. confirming a
day's forecast never changes if later test-period actuals are corrupted).

## Notes on modelling choices

A few decisions are documented in more depth in `reports/report.md` and in
code comments where they're made, but worth summarising here:

- **Rolling, not single-shot, evaluation.** The test period (final 14
  days) is split into 14 daily origins; each forecasts the next 24 hours
  from an expanding history, mirroring how a real system would re-forecast
  each morning. See `report.md`, Section 4, for the full reasoning and why
  this matters (concretely demonstrated in Section 9's day-by-day error
  analysis).
- **Conditional forecasts, clearly labelled.** SARIMAX-with-exog and the
  feature-based model both use real (not forecast) future weather values,
  which the assignment brief explicitly allows provided it's labelled as
  such — see Part 9, Q5 in the report.
- **Single dataset.** The assignment brief names and links only the one
  UCI dataset throughout; a note in `report.md` addresses this against the
  marking rubric's "both datasets" phrasing.
