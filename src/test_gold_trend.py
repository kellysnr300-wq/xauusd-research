"""Tests for strategies/gold_trend.py.

Usage:
    python -m src.test_gold_trend
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtester import Backtester
from src.lookahead import check_no_lookahead
from src.timeframes import max_entry_gap_for, resample_ohlc

spec = importlib.util.spec_from_file_location("gold_trend", Path("strategies/gold_trend.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GoldTrend = module.GoldTrend


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def daily(rows):
    times = pd.date_range("2024-01-01 22:00", periods=len(rows), freq="1D", tz="UTC")
    return pd.DataFrame({
        "timestamp": times,
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
    })


def flip(rows):
    return [(200 - o, 200 - l, 200 - h, 200 - c) for o, h, l, c in rows]


SMALL = dict(entry_n=3, exit_n=2, atr_n=2, stop_atr=2.0)
sigs = lambda rows, **p: list(GoldTrend(**{**SMALL, **p}).generate_signals(daily(rows)).signal)

# ---- hand-built: breakout, hold, channel exit -------------------------------------
FLAT = [(100, 101, 99, 100)] * 5
UP = FLAT + [
    (100, 103, 100, 102.5),    # 5: close above the 3-candle high (101) -> long
    (102.5, 104, 102, 103.5),  # 6: hold
    (103.5, 105, 103, 104.5),  # 7: hold
    (104.5, 105, 100, 100.5),  # 8: close below the 2-candle low (102) -> exit
    (100.5, 101, 100, 100.5),  # 9
]
check("long on the breakout candle, held, exited at the channel",
      sigs(UP) == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
check("nothing before the channel has enough history", sigs(UP)[:3] == [0, 0, 0])
check("short is the exact mirror", sigs(flip(UP)) == [0, 0, 0, 0, 0, -1, -1, -1, 0, 0])
check("longs can be disabled", sigs(UP, allow_long=False) == [0] * 10)
check("shorts can be disabled", sigs(flip(UP), allow_short=False) == [0] * 10)

# ---- the close-based ATR stop -------------------------------------------------------
# entry close 102.5, ATR 2.5 -> with stop_atr 1 the stop level is 100.0.
# The 6-candle exit channel still contains the 99 lows, so only the stop can
# fire when the close slips to 99.9.
FLAT9 = [(100, 101, 99, 100)] * 9
STOP = FLAT9 + [
    (100, 103, 100, 102.5), (102.5, 103, 101, 102), (102, 102.5, 100.5, 101),
    (101, 101.2, 99.8, 99.9), (99.9, 100, 99.5, 99.7),
]
BIG = dict(entry_n=8, exit_n=6, atr_n=2)
check("ATR stop exits when the close is 1 ATR below the entry",
      sigs(STOP, stop_atr=1.0, **BIG) == [0] * 9 + [1, 1, 1, 0, 0])
check("without the stop the same chart stays long",
      sigs(STOP, stop_atr=50.0, **BIG) == [0] * 9 + [1, 1, 1, 1, 1])

# ---- reversal and re-entry ---------------------------------------------------------------
REV = UP[:8] + [
    (104.5, 105, 96, 96.5),    # 8: closes below exit channel AND below the 3-candle low -> flip short
    (96.5, 97, 95, 95.5),      # 9: hold short
]
check("a break of the opposite channel reverses the position", sigs(REV)[7:] == [1, -1, -1])

# ---- trend filter -------------------------------------------------------------------------
check("trend filter lets a long through when the close is above the average",
      sigs(UP, trend_n=6)[5] == 1)
# a bounce inside a long decline: breaks the 3-candle high but stays under the 9-candle average
DECLINE = [(121 - 2 * i, 121 - 2 * i, 119 - 2 * i, 120 - 2 * i) for i in range(8)] + [(106, 113, 105, 112)]
check("without the filter the bounce is bought", sigs(DECLINE)[8] == 1)
check("trend filter blocks the same long below the average", sigs(DECLINE, trend_n=9)[8] == 0)

# ---- parameters and diagnostics -----------------------------------------------------------
for label, params in [("exit_n >= entry_n", {"entry_n": 20, "exit_n": 20}),
                      ("tiny windows", {"entry_n": 1}),
                      ("non-positive stop", {"stop_atr": 0})]:
    try:
        GoldTrend(**params); ok = False
    except ValueError:
        ok = True
    check(f"rejects {label}", ok)

strategy = GoldTrend(**SMALL)
strategy.generate_signals(daily(UP))
check("funnel counts the entry and the exit",
      strategy.diagnostics["1 long entries"] == 1 and strategy.diagnostics["3 exits at the channel"] == 1)

# ---- engine integration: trade at the NEXT open, no look-ahead ------------------------------
d = daily(UP)
trades = Backtester(d, GoldTrend(**SMALL), max_entry_gap=max_entry_gap_for("1d")).simulate()
check("one long trade", len(trades) == 1 and trades[0].side == "long")
check("entered at the OPEN after the breakout candle", trades[0].entry_price == d.open[6])
check("exited at the open after the exit signal", trades[0].exit_price == d.open[9])

# ---- look-ahead guard on realistic daily data ------------------------------------------------
def five_minute(years, seed, drift):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2018-01-01 22:00", periods=years * 52 * 7 * 288, freq="5min", tz="UTC")
    wd, hr = times.weekday, times.hour
    closed = ((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 22)) | (hr == 21)
    times = times[~closed]
    n = len(times)
    regime = np.repeat(rng.choice([-1.0, 1.0], size=n // 20000 + 1), 20000)[:n] * drift
    close = 2000 + (regime + rng.normal(0, 0.7, n)).cumsum()
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "timestamp": times, "open": open_,
        "high": np.maximum(open_, close) + rng.uniform(0, .4, n),
        "low": np.minimum(open_, close) - rng.uniform(0, .4, n),
        "close": close, "volume": 100,
    })


trending = resample_ohlc(five_minute(8, 4, drift=0.012), "1d")
random_walk = resample_ohlc(five_minute(8, 4, drift=0.0), "1d")
for label, params in [("defaults", {}), ("with trend filter", {"trend_n": 100}), ("faster", {"entry_n": 30, "exit_n": 10})]:
    problems = check_no_lookahead(lambda p=params: GoldTrend(**p), trending)
    check(f"no look-ahead: {label}", problems == [])

# ---- mechanics on synthetic regimes (NOT evidence of profit on real gold) ----------------------
def run(daily_data, **params):
    bt = Backtester(daily_data, GoldTrend(**params), max_entry_gap=max_entry_gap_for("1d"))
    trades = bt.simulate()
    return trades, sum(t.pnl for t in trades)

trades, total = run(trending)
check(f"trending market: profitable mechanics ({len(trades)} trades, {total:+.0f})", total > 0 and len(trades) >= 10)
shorts_only, total_short = run(trending, allow_long=False)
longs_only, total_long = run(trending, allow_short=False)
check("both sides contribute in a market that trends both ways",
      total_short != 0 and total_long != 0 and abs(total - (total_short + total_long)) < abs(total))
check("trades are infrequent (a handful per year, not hundreds)",
      len(trades) / 8 < 25)
_, total_rw = run(random_walk)
print(f"      info: random walk with no trend: {total_rw:+.0f} (expect about zero)")

print("\nAll gold trend tests passed.")
