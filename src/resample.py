import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
]


def resample_to_5m(data: pd.DataFrame) -> pd.DataFrame:
    """Convert 1-minute OHLCV data into 5-minute candles."""

    frame = data.copy()

    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]

    if "etc/utc" in frame.columns:
        frame = frame.rename(
            columns={"etc/utc": "timestamp"}
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="raise",
    )

    frame = frame.sort_values("timestamp")
    frame = frame.set_index("timestamp")

    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    if "volume" in frame.columns:
        aggregation["volume"] = "sum"

    result = frame.resample("5min").agg(aggregation)

    result = result.dropna(
        subset=["open", "high", "low", "close"]
    )

    result = result.reset_index()

    return result
