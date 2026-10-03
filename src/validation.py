import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
]


def validate_ohlc(data: pd.DataFrame) -> list[str]:
    """Validate basic OHLC candle integrity."""

    errors = []

    # Required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors

    # Timestamp checks
    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")

    if timestamps.isna().any():
        errors.append("Invalid timestamp values found.")

    if timestamps.duplicated().any():
        errors.append("Duplicate timestamps found.")

    if not timestamps.is_monotonic_increasing:
        errors.append("Timestamps are not sorted ascending.")

    # Numeric checks
    price_columns = ["open", "high", "low", "close"]

    for column in price_columns:
        values = pd.to_numeric(data[column], errors="coerce")

        if values.isna().any():
            errors.append(f"Invalid or missing values in {column}.")

        if (values <= 0).any():
            errors.append(f"Non-positive values found in {column}.")

    # OHLC relationship
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    open_ = pd.to_numeric(data["open"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")

    if ((high < open_) | (high < close) | (high < low)).any():
        errors.append("Invalid OHLC relationship: high is not the maximum.")

    if ((low > open_) | (low > close) | (low > high)).any():
        errors.append("Invalid OHLC relationship: low is not the minimum.")

    # Optional volume check
    if "volume" in data.columns:
        volume = pd.to_numeric(data["volume"], errors="coerce")

        if volume.isna().any():
            errors.append("Invalid or missing values in volume.")

        if (volume < 0).any():
            errors.append("Negative volume found.")

    return errors
