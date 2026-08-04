"""Feature-based ML model (XGBoost): fitting with hyperparameter tuning, and
a recursive rolling 24-hour-ahead forecast.

Rolling/recursive design
-------------------------
Same walk-forward evaluation as benchmarks.py and sarimax.py: 14 daily
origins across the test period, expanding history. The model itself is
fit ONCE (with hyperparameter tuning) and never refit during the rolling
loop -- only the *features* change at each step.

Within a single 24-hour forecast block, short lags (1, 2, 3, 6, 12) and the
rolling mean/std windows reference timestamps *inside that same block* for
hours after the first -- which aren't real yet. Per the brief's leakage
rules ("lagged and rolling features use only past observations"), those
values come from the model's OWN earlier predictions in the same block, not
from the real future. Concretely, the forecast proceeds one hour at a time
within each block: predict hour 1, treat that prediction as if it were
observed, use it to build hour 2's lag features, predict hour 2, and so on.
Lags >= 24 always reference a point outside the current block, so they're
always built from genuinely revealed actual data, never from a prediction.

Sensor/weather covariates for the forecast horizon are taken from their
REAL values in the test set -- a deliberate, brief-sanctioned choice ("if
realised future sensor or weather values are used from the test set, the
result should be described as a conditional forecast"). Forecasting the
weather itself is out of scope; this model's forecast should therefore be
read as conditional on realised future weather, not a true forecast made
from only information available at the origin. See report.md and Part 9 Q5.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from appliance_energy.features import (
    EXCLUDED_COLUMNS, DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS, add_time_features,
)

RANDOM_STATE = 0

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


def fit_feature_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 25,
    cv_splits: int = 3,
) -> XGBRegressor:
    """Fit an XGBoost regressor with hyperparameter tuning.

    Uses RandomizedSearchCV with TimeSeriesSplit (not a plain/shuffled K-fold
    -- the data is a time series, so folds must respect chronological order
    to avoid training on the future to predict the past during tuning).
    """
    base_model = XGBRegressor(
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_


def _build_feature_row(
    t: pd.Timestamp,
    working_target: pd.Series,
    raw_df: pd.DataFrame,
    target: str,
    feature_cols: list,
    lags: list,
    windows: list,
) -> pd.DataFrame:
    """Build a single-row feature vector for timestamp t.

    `working_target` holds actual values for all already-revealed
    timestamps, and the model's own predictions for any earlier hours in
    the current forecast block (see module docstring).
    """
    history = working_target.loc[:t].iloc[:-1]  # strictly before t

    row = {}

    # Sensor/weather covariates: real values at t (conditional forecast).
    for col in raw_df.columns:
        if col == target or col in EXCLUDED_COLUMNS:
            continue
        row[col] = raw_df.loc[t, col]

    # Time features: always known in advance.
    row["hour"] = t.hour
    row["dayofweek"] = t.dayofweek
    row["is_weekend"] = int(t.dayofweek >= 5)
    row["hour_sin"] = np.sin(2 * np.pi * t.hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * t.hour / 24)
    row["dow_sin"] = np.sin(2 * np.pi * t.dayofweek / 7)
    row["dow_cos"] = np.cos(2 * np.pi * t.dayofweek / 7)

    # Lag features.
    for lag in lags:
        idx = t - pd.Timedelta(hours=lag)
        row[f"lag_{lag}"] = history.get(idx, np.nan)

    # Rolling mean/std features (already shifted by construction of `history`).
    for window in windows:
        window_vals = history.iloc[-window:]
        row[f"roll_mean_{window}"] = window_vals.mean() if len(window_vals) else np.nan
        row[f"roll_std_{window}"] = window_vals.std() if len(window_vals) else np.nan

    return pd.DataFrame([row])[feature_cols]


def rolling_feature_forecast(
    model: XGBRegressor,
    raw_df: pd.DataFrame,
    train: pd.Series,
    test: pd.Series,
    feature_cols: list,
    target: str = "Appliances",
    lags: list = None,
    windows: list = None,
    block_size: int = 24,
) -> pd.Series:
    """Roll the fitted feature-based model forward across the test period.

    See module docstring for the recursive within-block prediction design.
    """
    if lags is None:
        lags = DEFAULT_LAGS
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS
    if len(test) % block_size != 0:
        raise ValueError(
            f"Test period length ({len(test)}) is not a multiple of "
            f"block_size ({block_size})."
        )

    # working_target: actual values for train, NaN placeholders for test
    # (filled in progressively as we predict, then overwritten with the
    # real actuals once each block is "revealed").
    working_target = pd.concat([train, pd.Series(np.nan, index=test.index)])

    n_blocks = len(test) // block_size
    predictions = []

    for k in range(n_blocks):
        block_index = test.index[k * block_size : (k + 1) * block_size]

        for t in block_index:
            row = _build_feature_row(
                t, working_target, raw_df, target, feature_cols, lags, windows
            )
            pred = float(model.predict(row)[0])
            working_target.loc[t] = pred  # pseudo-actual for later hours in this block
            predictions.append((t, pred))

        # Reveal this block's real actuals for use in future blocks.
        working_target.loc[block_index] = test.loc[block_index]

    forecast = pd.Series(
        [p for _, p in predictions], index=[t for t, _ in predictions], name="feature_model"
    )
    return forecast
