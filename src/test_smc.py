"""Tests for the SMC building blocks.

Usage:
    python -m src.test_smc
"""

import numpy as np
import pandas as pd

from src import smc
from src.timeframes import bar_end_times, resample_ohlc


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def frame(rows, start="2025-01-06", freq="5min"):
    times = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame({
        "timestamp": times,
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
    })


def mirror(rows):
    """Flip a chart upside down (valid OHLC kept)."""
    return [(200 - o, 200 - l, 200 - h, 200 - c) for o, h, l, c in rows]


def nan_eq(a, b):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


# ======================================================================
# Hand-built chart: bullish BOS at 9, retest of the OB at 13, CHoCH at 14
# ======================================================================
CHART = [
    (100, 101, 99, 100.5), (100.5, 102, 100, 101.5), (101.5, 105, 101, 104),
    (104, 104.5, 101.5, 102), (102, 102.5, 100, 100.5), (100.5, 101, 98, 98.5),
    (98.5, 99, 97, 97.5), (97.5, 101, 97.4, 100.5), (100.5, 104, 100.4, 103.5),
    (103.5, 106.5, 103, 106), (106, 107, 105, 106.5), (106.5, 106.8, 103, 104),
    (104, 104.5, 100, 100.5), (100.5, 100.9, 98.5, 99.5), (99.6, 100, 96, 96.5),
]
c = frame(CHART)

# ---- swings -----------------------------------------------------------------
sw = smc.swings(c, n=2)
check("swing high confirmed 2 candles late", bool(sw.swing_high[4]) and not sw.swing_high[:4].any())
check("swing high price", sw.swing_high_price[4] == 105 and sw.last_swing_high_idx[4] == 2)
check("swing low confirmed with price", bool(sw.swing_low[8]) and sw.swing_low_price[8] == 97)
check("last swing is NaN until confirmed", np.isnan(sw.last_swing_high[3]))

# ---- structure ----------------------------------------------------------------
st = smc.structure(c, n=2)
check("BOS up on first close above swing high", st.bos[9] == 1 and st.break_level[9] == 105 and st.break_swing_idx[9] == 2)
check("no event before the break", (st.bos[:9] == 0).all() and (st.choch[:9] == 0).all())
check("trend turns bullish at the BOS", (st.trend[:9] == 0).all() and (st.trend[9:14] == 1).all())
check("CHoCH down against the bullish trend", st.choch[14] == -1 and st.break_level[14] == 97 and st.trend[14] == -1)
check("a break is used only once", st.bos[10:14].sum() == 0)
wick = smc.structure(c, n=2, break_on="wick")
check("wick mode breaks on the wick", wick.bos[9] == 1 or wick.bos[8] == 1)
check("mirrored chart gives mirrored structure",
      smc.structure(frame(mirror(CHART)), n=2).bos[9] == -1
      and smc.structure(frame(mirror(CHART)), n=2).choch[14] == 1)

# ---- order blocks ---------------------------------------------------------------
ob = smc.order_blocks(c, n=2, lookback=3)
check("OB created on the break candle", bool(ob.ob_bull_new[9]) and ob.ob_bull_top[9] == 99 and ob.ob_bull_bottom[9] == 97)
check("OB is not testable on its own candle", not ob.ob_bull_retest[9])
check("retest of the OB", bool(ob.ob_bull_retest[13]) and ob.ob_bull_retest_top[13] == 99 and ob.ob_bull_retest_bottom[13] == 97)
check("OB used once", np.isnan(ob.ob_bull_top[13]))
check("bearish OB from the CHoCH", bool(ob.ob_bear_new[14]) and ob.ob_bear_top[14] == 107 and ob.ob_bear_bottom[14] == 105)
body = smc.order_blocks(c, n=2, lookback=3, zone="body")
check("body zone uses open/close", body.ob_bull_top[9] == 98.5 and body.ob_bull_bottom[9] == 97.5)
only_choch = smc.order_blocks(c, n=2, lookback=3, kinds="choch")
check("kinds='choch' skips the BOS block", not only_choch.ob_bull_new.any() and bool(only_choch.ob_bear_new[14]))
only_bos = smc.order_blocks(c, n=2, lookback=3, kinds="bos")
check("kinds='bos' skips the CHoCH block", bool(only_bos.ob_bull_new[9]) and not only_bos.ob_bear_new.any())
m = smc.order_blocks(frame(mirror(CHART)), n=2, lookback=3)
check("mirrored chart: bearish OB retest", bool(m.ob_bear_new[9]) and bool(m.ob_bear_retest[13]))

# ---- fair value gaps ----------------------------------------------------------------
FVG_UP = [
    (100, 101, 99, 100.5), (100.5, 104, 100.4, 103.5), (103.5, 106, 102, 105),
    (105, 105.5, 103, 104), (104, 104.5, 101.5, 102), (102, 102.5, 100.5, 101.8),
]
g = smc.fvg(frame(FVG_UP))
check("bullish FVG on the third candle", bool(g.fvg_bull_new[2]) and g.fvg_bull_top[2] == 102 and g.fvg_bull_bottom[2] == 101)
check("FVG not retested while price stays away", not g.fvg_bull_retest[3])
check("retest when price trades back in", bool(g.fvg_bull_retest[4]) and g.fvg_bull_retest_top[4] == 102)
check("FVG used once", not g.fvg_bull_retest[5] and np.isnan(g.fvg_bull_top[5]))
check("min_gap filters small gaps", not smc.fvg(frame(FVG_UP), min_gap=2).fvg_bull_new.any())
gm = smc.fvg(frame(mirror(FVG_UP)))
check("bearish FVG mirrors", bool(gm.fvg_bear_new[2]) and bool(gm.fvg_bear_retest[4]))
KILL = FVG_UP[:3] + [(105, 105.5, 100, 100.5)]
check("close through the gap invalidates it", not smc.fvg(frame(KILL)).fvg_bull_retest.any())

# ---- sweeps -------------------------------------------------------------------------
SW = [(99, 99.5, 98, 99), (99, 101, 98.5, 99.5), (99.5, 102, 99, 101.5), (101, 101.5, 99, 100.5)]
sp = smc.sweeps(frame(SW), np.full(4, 100.0), np.full(4, 100.0))
check("wick above and close back = high sweep", bool(sp.sweep_high[1]) and not sp.sweep_high[2])
check("wick below and close back = low sweep", bool(sp.sweep_low[3]) and not sp.sweep_low[1])
check("level of the previous close is used", not sp.sweep_high[0])

# ---- sessions -----------------------------------------------------------------------
stamps = pd.to_datetime([
    "2025-01-07 12:00", "2025-01-07 11:55",   # ny_am opens 07:00 EST
    "2025-01-07 07:00", "2025-01-07 06:55",   # london 02:00 EST
    "2025-01-07 09:55", "2025-01-07 10:00",
    "2025-01-07 01:00", "2025-01-07 04:55",   # asia 20:00-00:00 EST
    "2025-01-07 05:00", "2025-01-07 00:55",
    "2025-01-07 18:30", "2025-01-07 21:00",   # ny_pm 13:30-16:00 EST
    "2025-07-08 11:00", "2025-07-08 10:55",   # ny_am in summer (EDT)
], utc=True)
sess = smc.in_sessions(pd.DataFrame({"timestamp": stamps}))
expect = [
    ("ny_am", 0, True), ("ny_am", 1, False), ("london", 2, True), ("london", 3, False),
    ("london", 4, True), ("london", 5, False), ("asia", 6, True), ("asia", 7, True),
    ("asia", 8, False), ("asia", 9, False), ("ny_pm", 10, True), ("ny_pm", 11, False),
    ("ny_am", 12, True), ("ny_am", 13, False),
]
check("killzones follow New York time incl. daylight saving",
      all(bool(sess[name].iloc[i]) == want for name, i, want in expect))

days = pd.date_range("2025-01-07", periods=576, freq="5min", tz="UTC")
sd = pd.DataFrame({"timestamp": days, "open": 99.5, "high": 100.0, "low": 99.0, "close": 99.5})
def put(ts, col, value): sd.loc[sd.timestamp == pd.Timestamp(ts, tz="UTC"), col] = value
put("2025-01-07 03:00", "high", 110); put("2025-01-07 02:00", "low", 90)
put("2025-01-08 02:30", "high", 108); put("2025-01-08 03:30", "low", 92)
lv = smc.session_levels(sd, "asia")
at = lambda ts, col: lv.loc[sd.timestamp == pd.Timestamp(ts, tz="UTC"), col].iloc[0]
check("running session high", at("2025-01-07 03:00", "asia_high") == 110 and at("2025-01-07 02:55", "asia_high") == 100)
check("session values are NaN outside the session", np.isnan(at("2025-01-07 05:00", "asia_high")))
check("no previous session on day one", np.isnan(at("2025-01-07 03:00", "prev_asia_high")))
check("previous session range after it ends", at("2025-01-07 06:00", "prev_asia_high") == 110 and at("2025-01-07 06:00", "prev_asia_low") == 90)
check("previous range is kept during the next session", at("2025-01-08 02:30", "asia_high") == 108 and at("2025-01-08 02:30", "prev_asia_high") == 110)
check("previous range updates after the next session", at("2025-01-08 06:00", "prev_asia_high") == 108 and at("2025-01-08 06:00", "prev_asia_low") == 92)

# ---- daily / weekly levels ----------------------------------------------------------
hrs = pd.date_range("2025-01-05 22:00", periods=14 * 24, freq="1h", tz="UTC")
day_no = np.arange(len(hrs)) // 24
hour_no = np.arange(len(hrs)) % 24
hi = 100 + day_no + hour_no * 0.01
dl = pd.DataFrame({"timestamp": hrs, "open": hi - 0.1, "high": hi, "low": hi - 0.5, "close": hi - 0.2})
lev = smc.daily_levels(dl)
row = 3 * 24 + 5
check("previous day high", np.isclose(lev.pdh[row], 100 + 2 + 0.23))
check("previous day low", np.isclose(lev.pdl[row], 100 + 2 - 0.5))
check("previous day close", np.isclose(lev.pdc[row], 100 + 2 + 0.23 - 0.2))
check("no previous day on day one", np.isnan(lev.pdh[3]))
check("day high so far", np.isclose(lev.day_high[row], 100 + 3 + 0.05))
wk = 8 * 24 + 3
check("previous week high / low", np.isclose(lev.pwh[wk], 100 + 6 + 0.23) and np.isclose(lev.pwl[wk], 99.5))
check("no previous week in week one", np.isnan(lev.pwh[row]))

# ---- completed-candle times (daylight saving) --------------------------------------
ends = bar_end_times(pd.to_datetime(["2025-01-06 22:00", "2025-03-08 22:00", "2025-11-01 21:00"], utc=True), "1d")
check("daily candle is 24h in winter", ends[0] == pd.Timestamp("2025-01-07 22:00", tz="UTC"))
check("daily candle is 23h when clocks go forward", ends[1] == pd.Timestamp("2025-03-09 21:00", tz="UTC"))
check("daily candle is 25h when clocks go back", ends[2] == pd.Timestamp("2025-11-02 22:00", tz="UTC"))

# ---- multi-timeframe alignment --------------------------------------------------------
base = frame([(i, i + 1, i - 1, i) for i in range(2, 50)], start="2025-01-07 08:00")
h1 = resample_ohlc(base, "1h")
al = smc.align_htf(base, h1, "1h", ["close"])
ts = base.timestamp.dt.strftime("%H:%M")
get = lambda t: al.htf_close[ts == t].iloc[0]
check("no HTF value before the first hour finishes", np.isnan(get("08:50")))
check("hour is exposed when its last candle closes", get("08:55") == base.close[ts == "08:55"].iloc[0])
check("the unfinished hour is never exposed", get("10:50") == base.close[ts == "09:55"].iloc[0])
check("the new hour appears one candle later", get("10:55") == base.close[ts == "10:55"].iloc[0])
hourly = frame([(i, i + 1, i - 1, i) for i in range(2, 30)], start="2025-01-07 00:00", freq="1h")
al4 = smc.align_htf(hourly, resample_ohlc(hourly, "4h"), "4h", ["close"])
check("4h candle exposed after its last hourly candle",
      np.isnan(al4.htf_close[2]) and al4.htf_close[3] == hourly.close[3] and al4.htf_close[4] == hourly.close[3])

# ======================================================================
# The look-ahead guard: truncating the data must not change earlier output
# ======================================================================
def week_data(start, weeks, seed):
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=weeks * 7 * 288, freq="5min", tz="UTC")
    wd, hr = times.weekday, times.hour
    closed = ((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 22)) | (hr == 21)
    times = times[~closed]
    n = len(times)
    close = 2000 + rng.normal(0, 0.8, n).cumsum()
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "timestamp": times, "open": open_,
        "high": np.maximum(open_, close) + rng.uniform(0, .5, n),
        "low": np.minimum(open_, close) - rng.uniform(0, .5, n),
        "close": close, "volume": 100,
    })


def htf_features(d):
    h = resample_ohlc(d, "1h")
    h["trend"] = smc.structure(h, n=1).trend
    return smc.align_htf(d, h, "1h", ["trend", "close"])


BLOCKS = {
    "in_sessions": lambda d: smc.in_sessions(d),
    "session_levels": lambda d: smc.session_levels(d, "asia"),
    "daily_levels": smc.daily_levels,
    "swings": lambda d: smc.swings(d, n=2),
    "structure (close)": lambda d: smc.structure(d, n=2),
    "structure (wick)": lambda d: smc.structure(d, n=1, break_on="wick"),
    "fvg": lambda d: smc.fvg(d, min_gap=0.3),
    "order_blocks (range)": lambda d: smc.order_blocks(d, n=2),
    "order_blocks (body)": lambda d: smc.order_blocks(d, n=1, zone="body"),
    "sweeps of swings": lambda d: smc.sweeps(
        d, smc.swings(d, 2).last_swing_high, smc.swings(d, 2).last_swing_low),
    "sweeps of Asia range": lambda d: smc.sweeps(
        d, smc.session_levels(d, "asia").prev_asia_high,
        smc.session_levels(d, "asia").prev_asia_low),
    "align_htf with HTF structure": htf_features,
}

data = week_data("2025-01-06", weeks=3, seed=3)
for name, fn in BLOCKS.items():
    full = fn(data)
    ok = True
    for cut in (900, 1700, 2600):
        part = fn(data.iloc[:cut].reset_index(drop=True))
        for col in full.columns:
            if not nan_eq(full[col].iloc[:cut], part[col]):
                ok = False
    check(f"no look-ahead: {name}", ok)

check("blocks actually find things on random data",
      smc.structure(data, n=2).choch.ne(0).sum() > 20
      and smc.fvg(data, min_gap=0.3).fvg_bull_retest.sum() > 20
      and smc.order_blocks(data, n=2).ob_bull_retest.sum() > 5)

print("\nAll SMC tests passed.")
