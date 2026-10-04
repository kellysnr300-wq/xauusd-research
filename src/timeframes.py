"""Build higher-timeframe candles from the processed 5-minute data.

Conventions
-----------
- Candles are labelled by their OPEN time (UTC), like the source data.
- 15m / 30m / 1h / 4h bars align to the UTC clock (e.g. 4h = 00,04,08...).
- 1d bars follow the FX / gold convention: 17:00 to 17:00 New York time,
  so they stay correct across daylight-saving changes.
- Open = first, high = max, low = min, close = last, volume = sum.
- Periods with no data (weekends, holidays) produce no candle.
- A strategy only ever sees a finished candle, and its signal is traded
  at the NEXT candle's open, so higher timeframes add no look-ahead.
"""

from zoneinfo import ZoneInfoNotFoundError

import pandas as pd

BASE_TIMEFRAME = "5m"
DAY_ZONE = "America/New_York"
DAY_START_HOUR = 17  # New York wall clock

TIMEFRAMES = {
    "5m": {"minutes": 5, "rule": None, "label": "5 minutes (native)"},
    "15m": {"minutes": 15, "rule": "15min", "label": "15 minutes"},
    "30m": {"minutes": 30, "rule": "30min", "label": "30 minutes"},
    "1h": {"minutes": 60, "rule": "1h", "label": "1 hour"},
    "4h": {"minutes": 240, "rule": "4h", "label": "4 hours"},
    "1d": {"minutes": 1440, "rule": "1D", "label": "1 day"},
}

REQUIRED = ["timestamp", "open", "high", "low", "close"]


def check_timeframe(timeframe: str) -> str:
    if timeframe not in TIMEFRAMES:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. "
            f"Choose one of: {', '.join(TIMEFRAMES)}."
        )
    return timeframe


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=TIMEFRAMES[check_timeframe(timeframe)]["minutes"])


def max_entry_gap_for(timeframe: str) -> pd.Timedelta:
    """Largest gap between candles that still allows an entry fill.

    Six candles (never less than 30 minutes): quiet-hour holes are fine,
    weekend / holiday closures on intraday data are not.
    """
    return max(pd.Timedelta(minutes=30), 6 * timeframe_delta(timeframe))


def resample_ohlc(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 5-minute candles (UTC timestamps) to `timeframe`."""

    check_timeframe(timeframe)

    missing = [c for c in REQUIRED if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if timeframe == BASE_TIMEFRAME:
        return data.reset_index(drop=True)

    frame = data.sort_values("timestamp").set_index("timestamp")

    aggregation = {
        "open": "first", "high": "max", "low": "min", "close": "last",
    }
    if "volume" in frame.columns:
        aggregation["volume"] = "sum"
    frame = frame[list(aggregation)]

    if timeframe == "1d":
        # Trading day = 17:00 to 17:00 New York wall-clock time. Shifting
        # the wall clock by 7 hours makes each session fall on one
        # calendar date, which also handles 23h / 25h daylight-saving days.
        try:
            wall = frame.index.tz_convert(DAY_ZONE).tz_localize(None)
        except ZoneInfoNotFoundError as error:
            raise ValueError(
                "Daily candles need the timezone database. "
                "Install it with: pip install tzdata"
            ) from error
        session = (wall + pd.Timedelta(hours=24 - DAY_START_HOUR)).floor("D")
        result = frame.groupby(session.to_numpy(), sort=True).agg(aggregation)
        opened = pd.DatetimeIndex(result.index) - pd.Timedelta(
            hours=24 - DAY_START_HOUR)
        result.index = opened.tz_localize(DAY_ZONE).tz_convert("UTC")
    else:
        result = frame.resample(
            TIMEFRAMES[timeframe]["rule"], label="left", closed="left"
        ).agg(aggregation)

    result = result.dropna(subset=["open", "high", "low", "close"])
    result.index.name = "timestamp"
    return result.reset_index()
