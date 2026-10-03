"""Load the development dataset (never the out-of-sample period)."""

import pandas as pd

from src.paths import OOS_START
from src.validate_dataset import load_processed_files
from src.validation import validate_ohlc


def load_development(first_year: int, last_year: int) -> pd.DataFrame:
    """Load processed 5-minute candles for first_year..last_year."""

    if first_year > last_year:
        raise ValueError("first_year must be <= last_year")

    if last_year >= OOS_START.year:
        raise ValueError(
            f"Year {last_year} is in the out-of-sample period "
            f"(>= {OOS_START}). Refusing to load it."
        )

    frames = []

    for year in range(first_year, last_year + 1):
        try:
            frames.append(load_processed_files(str(year)))
        except FileNotFoundError:
            print(f"WARNING: no processed data for {year}")

    if not frames:
        raise FileNotFoundError("No processed data found.")

    data = pd.concat(frames, ignore_index=True)
    data = data.drop(columns=["source_file"], errors="ignore")
    data = data.sort_values("timestamp").reset_index(drop=True)

    errors = validate_ohlc(data)
    if errors:
        raise ValueError(f"Dataset failed validation: {errors}")

    return data
