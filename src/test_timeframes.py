"""Tests for the timeframe resampler.

Usage:
    python -m src.test_timeframes
"""

import numpy as np
import pandas as pd

from src.backtester import Backtester
from src.timeframes import (
    TIMEFRAMES, max_entry_gap_for, resample_ohlc, timeframe_delta,
)


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def week_data(start, weeks=1, seed=0):
    """5m candles with a realistic weekly pattern: closed Fri 22:00 UTC to
    Sun 22:00 UTC; plus a daily 1-hour break at 21:00-22:00 UTC."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=weeks * 7 * 288, freq="5min", tz="UTC")
    wd, hr = times.weekday, times.hour
    closed = ((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 22))
    closed |= (hr == 21)
    times = times[~closed]
    n = len(times)
    close = 2000 + rng.normal(0, 0.5, n).cumsum()
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "timestamp": times, "open": open_,
        "high": np.maximum(open_, close) + rng.uniform(0, .3, n),
        "low": np.minimum(open_, close) - rng.uniform(0, .3, n),
        "close": close, "volume": rng.integers(10, 100, n),
    })


d = week_data("2025-01-06", weeks=2)

# ---- aggregation correctness ------------------------------------------------
h = resample_ohlc(d, "1h")
first_hour = d[(d.timestamp >= h.timestamp.iloc[0]) & (d.timestamp < h.timestamp.iloc[0] + pd.Timedelta(hours=1))]
row = h.iloc[0]
check("1h open = first open", row.open == first_hour.open.iloc[0])
check("1h close = last close", row.close == first_hour.close.iloc[-1])
check("1h high/low = extremes", row.high == first_hour.high.max() and row.low == first_hour.low.min())
check("1h volume = sum", row.volume == first_hour.volume.sum())
check("1h bins start on the hour", (h.timestamp.dt.minute == 0).all())
check("15m bins on the quarter", (resample_ohlc(d, "15m").timestamp.dt.minute % 15 == 0).all())
check("4h bins on 00/04/08...", (resample_ohlc(d, "4h").timestamp.dt.hour % 4 == 0).all())

# ---- invariants for every timeframe ----------------------------------------
for tf in TIMEFRAMES:
    r = resample_ohlc(d, tf)
    ok = (
        r.high.max() == d.high.max()
        and r.low.min() == d.low.min()
        and r.volume.sum() == d.volume.sum()
        and r.timestamp.is_monotonic_increasing
        and not r.isna().any().any()
        and (r.high >= r[["open", "close"]].max(axis=1)).all()
        and (r.low <= r[["open", "close"]].min(axis=1)).all()
    )
    check(f"{tf}: extremes, volume, order, no NaN, OHLC valid", ok)

check("no candles during weekend closure",
      not resample_ohlc(d, "1h").timestamp.dt.weekday.isin([5]).any())
check("5m returns the data unchanged", resample_ohlc(d, "5m").equals(d.reset_index(drop=True)))
check("fewer candles at higher timeframes",
      len(resample_ohlc(d, "1h")) < len(resample_ohlc(d, "15m")) < len(d))

# ---- completed bars never change when later data arrives ---------------------
cut = d.iloc[: len(d) // 2]
full_h, part_h = resample_ohlc(d, "1h"), resample_ohlc(cut, "1h")
check("completed bars identical on truncated data (no look-ahead)",
      part_h.iloc[:-1].reset_index(drop=True).equals(full_h.iloc[: len(part_h) - 1].reset_index(drop=True)))

# ---- daily bars: 17:00 New York, across daylight saving ---------------------
def daily_start_hours(start):
    x = week_data(start, weeks=1)
    return sorted(set(resample_ohlc(x, "1d").timestamp.dt.hour))

check("daily bars start 22:00 UTC in winter", daily_start_hours("2025-01-06") == [22])
check("daily bars start 21:00 UTC in summer", daily_start_hours("2025-07-07") == [21])
year = week_data("2025-01-06", weeks=52)
daily = resample_ohlc(year, "1d")
check("about 5 daily bars per week", 250 <= len(daily) <= 262)
check("daily volume conserved", daily.volume.sum() == year.volume.sum())

# ---- errors and gap rule --------------------------------------------------------
try:
    resample_ohlc(d, "7m"); ok = False
except ValueError as e:
    ok = "Unknown timeframe" in str(e)
check("unknown timeframe rejected", ok)
check("entry gap scales with timeframe",
      max_entry_gap_for("5m") == pd.Timedelta(minutes=30)
      and max_entry_gap_for("1h") == pd.Timedelta(hours=6)
      and max_entry_gap_for("4h") == pd.Timedelta(hours=24))

# ---- engine integration: hourly entries must fill ----------------------------
class Always:
    def generate_signals(self, data):
        return [1, 0] * (len(data) // 2) + [0] * (len(data) % 2)

h = resample_ohlc(week_data("2025-01-06", weeks=2), "1h")
default = Backtester(h, Always()).simulate()
scaled = Backtester(h, Always(), max_entry_gap=max_entry_gap_for("1h")).simulate()
check("default 30-minute gap blocks hourly entries", len(default) == 0)
check("scaled gap lets hourly entries fill", len(scaled) > 10)

print("\nAll timeframe tests passed.")
