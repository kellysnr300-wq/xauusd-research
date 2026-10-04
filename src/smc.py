"""Smart-money-concept building blocks.

Every function takes the candle DataFrame (timestamp, open, high, low,
close[, volume]) and returns a DataFrame aligned to it. The rule for every
column: the value on row i is KNOWN AT THE CLOSE OF CANDLE i (it uses only
candles 0..i). A strategy that reads row i may therefore enter at the next
candle's open without look-ahead. tests (src/test_smc.py) check this for
every block by truncating the data and comparing outputs.

Definitions (defaults follow the common ICT / SMC teaching)
-----------------------------------------------------------
swing        fractal: higher/lower than `n` candles on each side. Known `n`
             candles after it forms (the confirmation candle).
BOS          first close beyond the latest swing in the direction of the
             current trend (trend 0 -> the first break sets the trend).
CHoCH        first close beyond the latest swing AGAINST the current trend;
             the trend flips.
FVG          three-candle gap. Bullish: low[i] > high[i-2]; bearish:
             high[i] < low[i-2]. Known at candle i (the third candle).
order block  the last opposite-coloured candle at the origin of the move
             that broke structure (bullish break -> last bearish candle).
zone life    a zone is retested when price trades back into it, and
             invalidated when a candle CLOSES through its far side. Each
             zone is used once.
sessions     killzones in New York time (daylight saving handled):
             asia 20:00-00:00, london 02:00-05:00, ny_am 07:00-10:00,
             ny_pm 13:30-16:00 (by candle open time). All editable.
"""

import numpy as np
import pandas as pd

from src.timeframes import (
    DAY_START_HOUR,
    DAY_ZONE,
    bar_end_times,
    check_timeframe,
)

NEW_YORK = "America/New_York"

KILLZONES = {
    "asia": ("20:00", "00:00"),
    "london": ("02:00", "05:00"),
    "ny_am": ("07:00", "10:00"),
    "ny_pm": ("13:30", "16:00"),
}


def _arrays(data: pd.DataFrame):
    return (
        data["open"].to_numpy(dtype=float),
        data["high"].to_numpy(dtype=float),
        data["low"].to_numpy(dtype=float),
        data["close"].to_numpy(dtype=float),
    )


# ----------------------------------------------------------------------
# Sessions / killzones
# ----------------------------------------------------------------------

def _minutes(text: str) -> int:
    hours, minutes = text.split(":")
    return int(hours) * 60 + int(minutes)


def in_sessions(data, windows=None, zone=NEW_YORK) -> pd.DataFrame:
    """Boolean column per window: is the candle (by open time) inside it?"""

    windows = windows or KILLZONES
    local = pd.DatetimeIndex(data["timestamp"]).tz_convert(zone)
    minute = (local.hour * 60 + local.minute).to_numpy()

    out = {}
    for name, (start_text, end_text) in windows.items():
        start, end = _minutes(start_text), _minutes(end_text)
        if start < end:
            out[name] = (minute >= start) & (minute < end)
        else:  # wraps midnight
            out[name] = (minute >= start) | (minute < end)

    return pd.DataFrame(out, index=data.index)


def session_levels(data, name, windows=None, zone=NEW_YORK) -> pd.DataFrame:
    """High / low of a session.

    {name}_high, {name}_low        running extremes while the session is
                                   open (NaN outside it)
    prev_{name}_high, prev_{name}_low   extremes of the last COMPLETED
                                   session (e.g. the Asia range, for sweeps)
    """

    windows = windows or KILLZONES
    flag = in_sessions(data, {name: windows[name]}, zone)[name].to_numpy()
    _, high, low, _ = _arrays(data)

    started = flag & ~np.concatenate([[False], flag[:-1]])
    instance = np.cumsum(started)

    s_high = pd.Series(np.where(flag, high, np.nan))
    s_low = pd.Series(np.where(flag, low, np.nan))

    running_high = s_high.groupby(instance).cummax().to_numpy()
    running_low = s_low.groupby(instance).cummin().to_numpy()
    final_high = s_high.groupby(instance).max()
    final_low = s_low.groupby(instance).min()

    # In a session the last completed one is the previous instance;
    # outside, it is the latest instance.
    ref = np.where(flag, instance - 1, instance)

    return pd.DataFrame({
        f"{name}_high": running_high,
        f"{name}_low": running_low,
        f"prev_{name}_high": final_high.reindex(ref).to_numpy(),
        f"prev_{name}_low": final_low.reindex(ref).to_numpy(),
    }, index=data.index)


# ----------------------------------------------------------------------
# Daily / weekly levels
# ----------------------------------------------------------------------

def daily_levels(data) -> pd.DataFrame:
    """Previous day / week levels (trading day = 17:00-17:00 New York).

    pdh, pdl, pdc            previous day high / low / close
    pwh, pwl                 previous week high / low
    day_high, day_low        the current day so far
    week_high, week_low      the current week so far
    """

    wall = pd.DatetimeIndex(data["timestamp"]).tz_convert(DAY_ZONE).tz_localize(None)
    day = (wall + pd.Timedelta(hours=24 - DAY_START_HOUR)).floor("D")
    week = day - pd.to_timedelta(day.weekday, unit="D")

    _, high, low, close = _arrays(data)
    out = pd.DataFrame(index=data.index)

    for key, names in ((day, ("pdh", "pdl", "pdc", "day")),
                       (week, ("pwh", "pwl", None, "week"))):
        codes, _ = pd.factorize(key, sort=True)
        frame = pd.DataFrame({"h": high, "l": low, "c": close, "code": codes})
        group = frame.groupby("code")
        g_high, g_low, g_close = group["h"].max(), group["l"].min(), group["c"].last()

        def previous(series):
            return series.shift(1).reindex(codes).to_numpy()

        out[names[0]] = previous(g_high)
        out[names[1]] = previous(g_low)
        if names[2]:
            out[names[2]] = previous(g_close)
        out[f"{names[3]}_high"] = group["h"].cummax().to_numpy()
        out[f"{names[3]}_low"] = group["l"].cummin().to_numpy()

    return out


# ----------------------------------------------------------------------
# Swings and market structure
# ----------------------------------------------------------------------

def swings(data, n: int = 1) -> pd.DataFrame:
    """Fractal swing highs / lows, flagged on their CONFIRMATION candle."""

    if n < 1:
        raise ValueError("n must be at least 1")

    _, high, low, _ = _arrays(data)
    h, l = pd.Series(high), pd.Series(low)

    left_h = h.shift(1).rolling(n).max()
    right_h = h[::-1].shift(1).rolling(n).max()[::-1]
    is_high = (h > left_h) & (h >= right_h)

    left_l = l.shift(1).rolling(n).min()
    right_l = l[::-1].shift(1).rolling(n).min()[::-1]
    is_low = (l < left_l) & (l <= right_l)

    conf_high = is_high.shift(n, fill_value=False).to_numpy(dtype=bool)
    conf_low = is_low.shift(n, fill_value=False).to_numpy(dtype=bool)

    position = np.arange(len(data), dtype=float) - n

    out = pd.DataFrame(index=data.index)
    out["swing_high"] = conf_high
    out["swing_low"] = conf_low
    out["swing_high_price"] = h.shift(n).where(conf_high).to_numpy()
    out["swing_low_price"] = l.shift(n).where(conf_low).to_numpy()
    out["last_swing_high"] = out["swing_high_price"].ffill()
    out["last_swing_low"] = out["swing_low_price"].ffill()
    out["last_swing_high_idx"] = pd.Series(
        np.where(conf_high, position, np.nan)).ffill().to_numpy()
    out["last_swing_low_idx"] = pd.Series(
        np.where(conf_low, position, np.nan)).ffill().to_numpy()
    return out


def structure(data, n: int = 1, break_on: str = "close") -> pd.DataFrame:
    """Trend, BOS and CHoCH events.

    trend           +1 bullish, -1 bearish, 0 unknown yet
    bos             +1 / -1 on the candle that confirms a break WITH the trend
    choch           +1 / -1 on the candle that breaks AGAINST the trend
    break_level     the swing price that was broken (NaN otherwise)
    break_swing_idx row position of the candle that formed that swing
    break_on        "close" (default) or "wick"
    """

    if break_on not in ("close", "wick"):
        raise ValueError("break_on must be 'close' or 'wick'")

    sw = swings(data, n)
    open_, high, low, close = _arrays(data)
    count = len(data)

    conf_high = sw["swing_high"].to_numpy()
    conf_low = sw["swing_low"].to_numpy()

    trend_out = np.zeros(count, dtype=int)
    bos = np.zeros(count, dtype=int)
    choch = np.zeros(count, dtype=int)
    level_out = np.full(count, np.nan)
    idx_out = np.full(count, np.nan)

    level_h = level_l = np.nan
    idx_h = idx_l = -1
    used_h = used_l = True
    trend = 0

    up_price = close if break_on == "close" else high
    down_price = close if break_on == "close" else low

    for j in range(count):
        if conf_high[j]:
            level_h, idx_h, used_h = high[j - n], j - n, False
        if conf_low[j]:
            level_l, idx_l, used_l = low[j - n], j - n, False

        up = (not used_h) and up_price[j] > level_h
        down = (not used_l) and down_price[j] < level_l

        if up and down:  # only possible with wick breaks
            if close[j] >= open_[j]:
                down = False
            else:
                up = False

        if up:
            used_h = True
            if trend < 0:
                choch[j] = 1
            else:
                bos[j] = 1
            trend = 1
            level_out[j], idx_out[j] = level_h, idx_h
        elif down:
            used_l = True
            if trend > 0:
                choch[j] = -1
            else:
                bos[j] = -1
            trend = -1
            level_out[j], idx_out[j] = level_l, idx_l

        trend_out[j] = trend

    return pd.DataFrame({
        "trend": trend_out, "bos": bos, "choch": choch,
        "break_level": level_out, "break_swing_idx": idx_out,
    }, index=data.index)


# ----------------------------------------------------------------------
# Zones: fair value gaps and order blocks
# ----------------------------------------------------------------------

def _track_zones(count, creations, high, low, close, max_age, max_zones):
    """Follow zones through time; shared by fvg() and order_blocks().

    creations: {candle: [(side, top, bottom), ...]}, side +1 bull / -1 bear.
    A zone created on candle j becomes testable from candle j + 1.
    """

    names = ("top", "bottom", "retest_top", "retest_bottom")
    out = {f"{side}_{n}": np.full(count, np.nan)
           for side in ("bull", "bear") for n in names}
    for side in ("bull", "bear"):
        out[f"{side}_new"] = np.zeros(count, dtype=bool)
        out[f"{side}_retest"] = np.zeros(count, dtype=bool)

    bull, bear = [], []  # zones: (top, bottom, born)

    for j in range(count):
        if bull:
            keep, hit = [], None
            for zone in bull:
                top, bottom, born = zone
                if j - born > max_age or close[j] < bottom:
                    continue
                if low[j] <= top:
                    if hit is None or top > hit[0]:
                        hit = zone
                    continue
                keep.append(zone)
            bull = keep
            if hit is not None:
                out["bull_retest"][j] = True
                out["bull_retest_top"][j], out["bull_retest_bottom"][j] = hit[0], hit[1]

        if bear:
            keep, hit = [], None
            for zone in bear:
                top, bottom, born = zone
                if j - born > max_age or close[j] > top:
                    continue
                if high[j] >= bottom:
                    if hit is None or bottom < hit[1]:
                        hit = zone
                    continue
                keep.append(zone)
            bear = keep
            if hit is not None:
                out["bear_retest"][j] = True
                out["bear_retest_top"][j], out["bear_retest_bottom"][j] = hit[0], hit[1]

        for side, top, bottom in creations.get(j, ()):
            if side > 0:
                bull = (bull + [(top, bottom, j)])[-max_zones:]
                out["bull_new"][j] = True
            else:
                bear = (bear + [(top, bottom, j)])[-max_zones:]
                out["bear_new"][j] = True

        if bull:
            out["bull_top"][j], out["bull_bottom"][j] = bull[-1][0], bull[-1][1]
        if bear:
            out["bear_top"][j], out["bear_bottom"][j] = bear[-1][0], bear[-1][1]

    return out


def _zone_frame(prefix, out, index):
    return pd.DataFrame(
        {f"{prefix}_{key}": value for key, value in out.items()}, index=index)


def fvg(data, min_gap: float = 0.0, max_age: int = 500,
        max_zones: int = 20) -> pd.DataFrame:
    """Fair value gaps and their retests.

    fvg_bull_new / fvg_bear_new        a gap formed on this candle
    fvg_bull_top / _bottom             latest still-open bullish gap
    fvg_bull_retest                    this candle traded back into an open
                                       bullish gap (edges in *_retest_top /
                                       *_retest_bottom); bearish mirrored
    """

    _, high, low, close = _arrays(data)
    count = len(data)
    creations = {}

    for j in range(2, count):
        if low[j] > high[j - 2] and low[j] - high[j - 2] >= min_gap:
            creations.setdefault(j, []).append((1, low[j], high[j - 2]))
        elif high[j] < low[j - 2] and low[j - 2] - high[j] >= min_gap:
            creations.setdefault(j, []).append((-1, low[j - 2], high[j]))

    out = _track_zones(count, creations, high, low, close, max_age, max_zones)
    return _zone_frame("fvg", out, data.index)


def order_blocks(data, n: int = 1, break_on: str = "close",
                 lookback: int = 5, zone: str = "range",
                 kinds: str = "both", max_age: int = 500,
                 max_zones: int = 20) -> pd.DataFrame:
    """Order blocks created by structure breaks, and their retests.

    Same columns as fvg() with the prefix "ob". A bullish break creates a
    bullish OB: the last bearish candle at or before the lowest low since
    the broken swing (within `lookback` candles). Bearish mirrored.

    zone   "range" = full candle (high/low), "body" = open/close only
    kinds  "both", "bos" or "choch": which breaks create order blocks
    """

    if zone not in ("range", "body"):
        raise ValueError("zone must be 'range' or 'body'")
    if kinds not in ("both", "bos", "choch"):
        raise ValueError("kinds must be 'both', 'bos' or 'choch'")

    st = structure(data, n, break_on)
    open_, high, low, close = _arrays(data)
    count = len(data)

    bos = st["bos"].to_numpy()
    choch = st["choch"].to_numpy()
    swing_idx = st["break_swing_idx"].to_numpy()
    creations = {}

    for j in np.flatnonzero((bos != 0) | (choch != 0)):
        is_choch = choch[j] != 0
        if (kinds == "bos" and is_choch) or (kinds == "choch" and not is_choch):
            continue

        side = int(choch[j] or bos[j])
        start = int(swing_idx[j])

        if side > 0:
            origin = start + int(np.argmin(low[start: j + 1]))
        else:
            origin = start + int(np.argmax(high[start: j + 1]))

        block = None
        for k in range(origin, max(origin - lookback, -1), -1):
            if (side > 0 and close[k] < open_[k]) or \
               (side < 0 and close[k] > open_[k]):
                block = k
                break
        if block is None:
            continue

        if zone == "range":
            top, bottom = high[block], low[block]
        else:
            top = max(open_[block], close[block])
            bottom = min(open_[block], close[block])

        creations.setdefault(int(j), []).append((side, top, bottom))

    out = _track_zones(count, creations, high, low, close, max_age, max_zones)
    return _zone_frame("ob", out, data.index)


# ----------------------------------------------------------------------
# Liquidity sweeps
# ----------------------------------------------------------------------

def sweeps(data, high_level, low_level) -> pd.DataFrame:
    """Wick through a level, close back inside it.

    high_level / low_level are Series of levels (e.g. last_swing_high,
    prev_asia_high, pdh). The level known at the PREVIOUS close is used, so
    the sweep candle cannot define its own level.

    sweep_high  high > level and close < level  (sweep of buy-side liquidity)
    sweep_low   low < level and close > level   (sweep of sell-side liquidity)
    """

    _, high, low, close = _arrays(data)
    level_h = pd.Series(np.asarray(high_level, dtype=float)).shift(1).to_numpy()
    level_l = pd.Series(np.asarray(low_level, dtype=float)).shift(1).to_numpy()

    sweep_high = (high > level_h) & (close < level_h)
    sweep_low = (low < level_l) & (close > level_l)

    return pd.DataFrame({
        "sweep_high": sweep_high,
        "sweep_low": sweep_low,
        "sweep_high_level": np.where(sweep_high, level_h, np.nan),
        "sweep_low_level": np.where(sweep_low, level_l, np.nan),
    }, index=data.index)


# ----------------------------------------------------------------------
# Multi-timeframe
# ----------------------------------------------------------------------

def align_htf(data, htf: pd.DataFrame, timeframe: str, columns,
              prefix: str = "htf_") -> pd.DataFrame:
    """Bring higher-timeframe values onto the base candles, without peeking.

    `htf` is a DataFrame with a `timestamp` column (candle open times), e.g.
    resample_ohlc(data, "1h") plus any columns you computed on it (trend,
    zones, ...). Each base candle gets the values of the last HTF candle
    that was already FINISHED when the base candle closed.
    """

    check_timeframe(timeframe)
    columns = list(columns)

    stamps = pd.DatetimeIndex(data["timestamp"])
    head = stamps[:200]
    steps = head[1:] - head[:-1]
    steps = steps[steps > pd.Timedelta(0)]
    base = steps.min() if len(steps) else pd.Timedelta(0)

    closes = (stamps + base).astype("datetime64[ns, UTC]")
    ends = bar_end_times(htf["timestamp"], timeframe).astype("datetime64[ns, UTC]")

    left = pd.DataFrame({"t": closes, "pos": np.arange(len(data))})
    right = htf[columns].copy()
    right["t"] = ends
    right = right.sort_values("t")

    merged = pd.merge_asof(left.sort_values("t"), right, on="t", direction="backward")
    merged = merged.sort_values("pos")

    result = merged[columns].copy()
    result.columns = [f"{prefix}{c}" for c in columns]
    result.index = data.index
    return result
