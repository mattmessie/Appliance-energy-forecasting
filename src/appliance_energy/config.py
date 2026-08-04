"""Project-wide configuration and paths."""

from pathlib import Path

RANDOM_STATE = 0

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "model_objects"

for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FORECAST_DIR, METRICS_DIR, FIGURE_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

TARGET = "Appliances"

# Hourly data: 24 obs = 1 day, 168 obs = 1 week
DAILY_PERIOD = 24
WEEKLY_PERIOD = 168

# Final 14 days as test set (hourly data)
TEST_STEPS = 14 * 24

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
