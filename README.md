# Appliance Energy Forecasting

This repository contains a reproducible time-series forecasting pipeline for modelling and forecasting household appliance energy use.

The project uses the **Appliances Energy Prediction** dataset, which contains appliance energy consumption, indoor temperature and humidity sensor measurements, outdoor weather variables, and timestamp information. The aim is to compare simple benchmark models, a SARIMAX model, a feature-based machine-learning model, and a time-series foundation model.

## Project aim

The aim of this assignment is to forecast short-term household appliance energy use and evaluate whether increasingly complex models improve on simple benchmark methods.

The main questions are:

1. How well do simple benchmark models forecast appliance energy use?
2. Does a SARIMAX model improve on the benchmark forecasts?
3. Do sensor, weather, and time-based covariates improve forecast accuracy?
4. Does a feature-based machine-learning model such as XGBoost improve performance?
5. Does a time-series foundation model such as Chronos, TimesFM, or TimeGPT provide any additional benefit?
6. Which model would be most suitable for a practical smart-home energy forecasting system?

## Dataset

The dataset used in this project is the **Appliances Energy Prediction** dataset.

The target variable is:

```text
Appliances
```

This represents household appliance energy use for each time interval.

The original dataset is sampled every 10 minutes and contains variables including:

```text
date
Appliances
lights
T1, RH_1
T2, RH_2
T3, RH_3
T4, RH_4
T5, RH_5
T6, RH_6
T7, RH_7
T8, RH_8
T9, RH_9
T_out
Press_mm_hg
RH_out
Windspeed
Visibility
Tdewpoint
```

The `T` variables are indoor temperature measurements from different rooms or sensor locations. The `RH` variables are indoor relative humidity measurements. The outdoor weather variables include outdoor temperature, pressure, outdoor humidity, wind speed, visibility, and dew point.

## Forecasting task

The main forecasting task is:

```text
Forecast appliance energy use over the next 24 hours.
```

If using the original 10-minute data, the 24-hour forecast horizon is:

```python
horizon = 24 * 6
horizon = 144
```

If the data are resampled to hourly averages, the 24-hour forecast horizon is:

```python
horizon = 24
```

For this assignment, students may resample the original 10-minute data to hourly data to make SARIMAX modelling and pipeline execution more manageable.

The recommended test period is the final 14 days of the dataset.

For 10-minute data:

```python
test_steps = 14 * 24 * 6
```

For hourly data:

```python
test_steps = 14 * 24
```

## Models

The project should compare the following model classes.

### 1. Benchmark models

Include several simple benchmark forecasts:

```text
Mean forecast
Naive forecast
Daily seasonal naive forecast
Weekly seasonal naive forecast
Drift forecast
```

For hourly data:

```text
Daily seasonal naive: same hour yesterday, lag 24
Weekly seasonal naive: same hour last week, lag 168
```

For 10-minute data:

```text
Daily seasonal naive: same time yesterday, lag 144
Weekly seasonal naive: same time last week, lag 1008
```

### 2. SARIMAX model

Fit a SARIMAX model to the appliance energy series.

For hourly data, a simple starting point is:

```python
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 24)
```

This captures short-term autocorrelation and daily seasonality.

Students may fit:

```text
Target-only SARIMA/SARIMAX
SARIMAX with exogenous variables
```

Possible exogenous variables include:

```text
T_out
RH_out
Windspeed
Visibility
Tdewpoint
hour_sin
hour_cos
dow_sin
dow_cos
```

### 3. Feature-based model

Fit a feature-based machine-learning model such as:

```text
XGBoost
LightGBM
Random Forest
HistGradientBoostingRegressor
```

The feature table should include:

```text
Lagged appliance energy use
Rolling means
Rolling standard deviations
Hour-of-day features
Day-of-week features
Weekend indicator
Indoor temperature variables
Indoor humidity variables
Outdoor weather variables
```

Students should ensure that lagged and rolling features use only past observations.

### 4. Foundation model

Fit a time-series foundation model such as:

```text
Chronos
TimesFM
TimeGPT
```

The foundation model may be used as:

```text
Target-only forecasting model
Covariate-informed forecasting model, if supported
Zero-shot model
Fine-tuned or adapted model, if appropriate
```

Students should clearly explain how the foundation model is being used and whether it has access to covariates.

## Feature sources

The project uses three main types of features.

### Original measured variables

These come directly from the dataset:

```text
Appliances
lights
indoor temperature variables
indoor humidity variables
outdoor weather variables
```

### Time-based features

These are created from the timestamp:

```text
hour
dayofweek
is_weekend
hour_sin
hour_cos
dow_sin
dow_cos
```

Example:

```python
data["hour"] = data.index.hour
data["dayofweek"] = data.index.dayofweek
data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7)
```

### Lag and rolling features

These are created from the target variable, `Appliances`.

For hourly data:

```python
data["lag_1"] = data["Appliances"].shift(1)
data["lag_24"] = data["Appliances"].shift(24)
data["lag_168"] = data["Appliances"].shift(168)

data["roll_mean_24"] = data["Appliances"].shift(1).rolling(24).mean()
data["roll_std_24"] = data["Appliances"].shift(1).rolling(24).std()
```

The `.shift(1)` is important because it prevents the model from using the current or future value of the target variable.

## Repository structure

A suggested structure is:

```text
appliance-energy-forecasting/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_download_and_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_benchmark_models.ipynb
│   ├── 04_sarimax_models.ipynb
│   ├── 05_feature_based_models.ipynb
│   ├── 06_foundation_model.ipynb
│   └── 07_model_comparison.ipynb
│
├── src/
│   └── appliance_energy/
│       ├── __init__.py
│       ├── config.py
│       ├── pipeline.py
│       ├── data.py
│       ├── features.py
│       ├── evaluation.py
│       ├── plotting.py
│       └── models/
│           ├── __init__.py
│           ├── benchmarks.py
│           ├── sarimax.py
│           ├── feature_models.py
│           └── foundation.py
│
├── scripts/
│   ├── download_data.py
│   ├── make_features.py
│   ├── run_pipeline.py
│   └── evaluate_models.py
│
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   ├── metrics/
│   └── model_objects/
│
├── reports/
│   ├── report.md
│   └── figures/
│
└── tests/
    ├── test_features.py
    ├── test_evaluation.py
    └── test_benchmarks.py
```

## Installation

Create a Python environment.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` should include:

```text
numpy
pandas
matplotlib
scikit-learn
statsmodels
```

If using XGBoost:

```text
xgboost
```

If using TimeGPT:

```text
nixtla
```

If using Chronos or TimesFM, additional packages such as `torch`, `transformers`, or model-specific dependencies may be required.

## Running the pipeline

The full analysis should be reproducible from the command line.

The main pipeline entry point should be:

```bash
python scripts/run_pipeline.py
```

The pipeline should:

1. Load or download the dataset.
2. Clean and prepare the time series.
3. Create time, lag, rolling, sensor, and weather features.
4. Split the data into training and test sets.
5. Fit benchmark models.
6. Fit the SARIMAX model.
7. Fit the feature-based model.
8. Fit or call the foundation model.
9. Evaluate all forecasts.
10. Save forecasts, metrics, and plots.

## Outputs

The pipeline should save forecasts to:

```text
outputs/forecasts/all_forecasts.csv
```

This file should contain the actual values and model forecasts:

```text
actual
mean
naive
seasonal_naive_daily
seasonal_naive_weekly
drift
sarimax
feature_model
foundation_model
```

The pipeline should save model comparison metrics to:

```text
outputs/metrics/model_comparison.csv
```

This file should contain:

```text
model
MAE
RMSE
MASE
Bias
```

The pipeline should save figures to:

```text
outputs/figures/
```

Suggested figures include:

```text
forecast_comparison.png
error_diagnostics.png
residual_acf.png
feature_importance.png
```

## Evaluation metrics

All models should be evaluated on the same test period.

Required metrics:

```text
MAE
RMSE
MASE
Bias
```

Students should compare all models against the strongest benchmark, not just against each other.

## Data leakage

Students must avoid data leakage.

Examples of leakage include:

```text
Using future values of Appliances in lag or rolling features
Creating rolling features without shifting the target first
Scaling the full dataset before the train-test split
Using future sensor or weather values without discussing forecast realism
Choosing the final model based only on test-set performance
```

Important point:

```text
Future time-of-day and day-of-week variables are known in advance. Future indoor sensor and weather variables may not be known in a real operational forecast. If realised future sensor or weather values are used from the test set, the result should be described as a conditional forecast.
```

## Report

The final report should be 6–8 pages and should describe the full analysis.

Suggested report structure:

```text
1. Introduction
2. Data and preprocessing
3. Exploratory analysis
4. Forecasting design
5. Benchmark models
6. SARIMAX model
7. Feature-based model
8. Foundation model
9. Results and error analysis
10. Discussion and limitations
11. Conclusion
```

The report should answer the following questions:

1. Which benchmark model is strongest, and what does this reveal about appliance energy use?
2. Does SARIMAX improve on the strongest benchmark?
3. Does the feature-based model improve when lag, rolling, time, sensor, and weather features are added?
4. Does the foundation model outperform the simpler models?
5. Which covariates would genuinely be known at the forecast origin?
6. Which model would you recommend for practical smart-home energy forecasting, and why?

## Tests

The repository should include simple tests for important functions.

Examples:

```text
test that forecast lengths match the test period
test that MASE is zero for a perfect forecast
test that lag features do not use future target values
test that the processed dataset has no missing target values
```

Run tests using:

```bash
pytest
```

## Good practice

Students should follow these principles:

```text
Use clear function names.
Keep reusable code in src/.
Keep notebooks for exploration and explanation.
Keep scripts small and focused.
Do not commit large raw data files.
Make the pipeline reproducible from a fresh clone.
Set random seeds where relevant.
Compare every advanced model against simple benchmarks.
Explain whether covariates are known at the forecast origin.
Document any modelling assumptions.
```

## Expected submission

The submitted repository should include:

```text
README.md
requirements.txt or environment.yml
source code in src/
pipeline script in scripts/
notebooks showing exploration and results
generated metrics and figures
final report
```

The repository should run from a fresh clone using the instructions in this README.
