from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load and normalize OHLC data from a CSV file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}"
        )

    data = pd.read_csv(path)

    # Normalize column names.
    data.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in data.columns
    ]

    # Dukascopy exports the timestamp column as "Etc/UTC".
    if "etc/utc" in data.columns:
        data = data.rename(
            columns={"etc/utc": "timestamp"}
        )

    if "timestamp" not in data.columns:
        raise ValueError(
            "CSV must contain a 'timestamp' or 'Etc/UTC' column."
        )

    # Convert timestamps to timezone-aware UTC.
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="coerce",
    )

    if data["timestamp"].isna().any():
        raise ValueError(
            "CSV contains invalid timestamps."
        )

    # Normalize OHLCV column names.
    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}"
        )

    # Convert numeric market-data columns.
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    if "volume" in data.columns:
        numeric_columns.append("volume")

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data[numeric_columns].isna().any().any():
        raise ValueError(
            "CSV contains invalid numeric market-data values."
        )

    # Sort chronologically.
    data = (
        data
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return data
