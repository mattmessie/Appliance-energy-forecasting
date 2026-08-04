"""Data loading and preparation for the appliance energy series."""

import pandas as pd

from appliance_energy.config import DATA_URL, RAW_DIR, PROCESSED_DIR, TARGET


def download_raw_data(force: bool = False) -> pd.DataFrame:
    """Download the raw 10-minute UCI Appliances Energy Prediction dataset.

    Caches a copy in data/raw/ so repeat runs don't re-hit the network.
    """
    raw_path = RAW_DIR / "energydata_complete.csv"

    if raw_path.exists() and not force:
        df = pd.read_csv(raw_path)
    else:
        print("Downloading data from UCI...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(raw_path, index=False)

    return df


def load_appliance_data(force_download: bool = False) -> pd.DataFrame:
    """Load, clean, and resample the appliance energy series to hourly.

    Steps:
      1. Download (or load cached) raw 10-minute data.
      2. Parse the timestamp and set it as a sorted DatetimeIndex.
      3. Coerce all columns to numeric, drop rows with a missing target.
      4. Resample 10-minute -> hourly using the mean.
      5. Interpolate any small gaps left by resampling, then drop remaining NaNs.
      6. Cache the processed hourly series.
    """
    df = download_raw_data(force=force_download)

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])

    print("Original (10-min) data shape:", df.shape)
    print("Date range:", df.index.min(), "to", df.index.max())

    hourly = df.resample("h").mean()
    hourly = hourly.interpolate("time")
    hourly = hourly.dropna()

    print("Hourly data shape:", hourly.shape)

    hourly.to_csv(PROCESSED_DIR / "appliance_hourly.csv")

    return hourly


if __name__ == "__main__":
    load_appliance_data()
