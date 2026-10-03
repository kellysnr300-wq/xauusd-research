from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLC data from a CSV file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    data = pd.read_csv(path)

    # Normalize column names
    data.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in data.columns
    ]

    if "timestamp" not in data.columns:
        raise ValueError("CSV must contain a 'timestamp' column.")

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    if data["timestamp"].isna().any():
        raise ValueError("CSV contains invalid timestamps.")

    data = data.sort_values("timestamp").reset_index(drop=True)

    return data
