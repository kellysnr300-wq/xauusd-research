"""Tests for strategies/smc_v1.py.

Usage:
    python -m src.test_smc_v1
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtester import Backtester
from src.lookahead import check_no_lookahead
from src.smc import align_htf, order_blocks, structure
from src.timeframes import resample_ohlc

spec = importlib.util.spec_from_file_location("smc_v1", Path("strategies/smc_v1.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SmcV1 = module.SmcV1


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def frame(rows, start):
    times = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": times,
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
    })


# Same hand-built chart as test_smc: BOS at candle 9, order block 97-99,
# retest at candle 13.
CHART = [
    (100, 101, 99, 100.5), (100.5, 102, 100, 101.5), (101.5, 105, 101, 104),
    (104, 104.5, 101.5, 102), (102, 102.5, 100, 100.5), (100.5, 101, 98, 98.5),
    (98.5, 99, 97, 97.5), (97.5, 101, 97.4, 100.5), (100.5, 104, 100.4, 103.5),
    (103.5, 106.5, 103, 106), (106, 107, 105, 106.5), (106.5, 106.8, 103, 104),
    (104, 104.5, 100, 100.5), (100.5, 100.9, 98.5, 99.5),
]
EXTRA = [(99.6, 100, 99, 99.8), (99.8, 109.5, 99.5, 109), (109, 110, 108, 109)]

BASE = dict(swing=2, lookback=3, buffer=0.5, min_risk=1.0, htf="none")
NY_AM = "2025-01-07 12:00"   # 07:00 New York
NIGHT = "2025-01-07 03:00"   # 22:00 New York, outside london / ny_am


def signals(rows, start, **params):
    strategy = SmcV1(**{**BASE, **params})
    out = strategy.generate_signals(frame(rows, start))
    return strategy, out


# ---- the hand-built setup -----------------------------------------------------
s, out = signals(CHART, NY_AM)
check("entry on the retest candle", list(out.index[out.entry != 0]) == [13])
check("long, stop beyond the block", out.entry[13] == 1 and out.stop_price[13] == 96.5)
check("target is 3R", out.target_r[13] == 3.0)
check("funnel counts the single setup", list(s.diagnostics.values()) == [1, 1, 1, 1, 1])

# ---- every filter explains itself ----------------------------------------------
s, out = signals(CHART, NIGHT)
check("outside the killzone: no entry", (out.entry == 0).all())
d = list(s.diagnostics.values())
check("funnel shows it died at the killzone", d[0] == 1 and d[1] == 0 and d[-1] == 0)
s, out = signals(CHART, NIGHT, killzones="")
check("killzone off: entry returns", out.entry[13] == 1)
s, out = signals(CHART, NY_AM, kinds="choch")
check("kinds='choch' removes the BOS block", (out.entry == 0).all())
s, out = signals(CHART, NY_AM, min_risk=5.0)
check("risk limits remove tiny stops", (out.entry == 0).all() and list(s.diagnostics.values())[-1] == 0)
s, out = signals(CHART, NY_AM, require_sweep=True, sweep_levels="swing")
check("sweep required but none happened", (out.entry == 0).all())
s, out = signals(CHART, NY_AM, allow_long=False)
check("longs can be disabled", (out.entry == 0).all())

# ---- end to end: exact 3R through the engine ------------------------------------------
d = frame(CHART + EXTRA, NY_AM)
bt = Backtester(d, SmcV1(**BASE))
trades = bt.simulate()
check("one trade", len(trades) == 1 and trades[0].side == "long")
check("take profit at exactly 3 x risk",
      trades[0].exit_reason == "take_profit"
      and abs(trades[0].pnl - 3 * (99.6 - 96.5)) < 1e-9)

# ---- silent zero is now reported -------------------------------------------------------
bt = Backtester(frame(CHART, NIGHT), SmcV1(**BASE))
bt.simulate()
check("zero entries produce a warning", any("no entries at all" in w for w in bt.warnings))

# ---- bad parameters fail loudly ---------------------------------------------------------
for label, params in [("killzone", {"killzones": "tokyo"}), ("htf", {"htf": "7m"}),
                      ("sweep level", {"require_sweep": True, "sweep_levels": "moon"})]:
    try:
        SmcV1(**params); ok = False
    except ValueError:
        ok = True
    check(f"unknown {label} rejected", ok)

# ---- random data: filters only ever remove trades, never add ----------------------------
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


data = week_data("2022-01-03", 12, 21)
cfg = dict(swing=2, max_risk=40.0)
entries = lambda **p: set(np.flatnonzero(SmcV1(**{**cfg, **p}).generate_signals(data).entry.to_numpy() != 0))

none = entries(killzones="", htf="none")
zone = entries(killzones="london,ny_am", htf="none")
bias = entries(killzones="", htf="1h")
both = entries(killzones="london,ny_am", htf="1h")
sweep = entries(killzones="", htf="none", require_sweep=True)
# A candle that retests a bullish AND a bearish zone is skipped without
# filters (the strategy will not pick a side) but a filter may pick one.
ob = order_blocks(data, n=2, lookback=5, zone="range", kinds="both", max_age=200)
conflict = set(np.flatnonzero((ob.ob_bull_retest & ob.ob_bear_retest).to_numpy()))
only_removes = lambda filtered, base: (filtered - base) <= conflict and len(filtered) < len(base)

check("random data produces setups", len(none) > 100)
check("killzone filter only removes", only_removes(zone, none))
check("HTF bias filter only removes", only_removes(bias, none))
check("filters combine", (both - zone) <= conflict and (both - bias) <= conflict and len(both) < len(zone))
check("sweep filter only removes", only_removes(sweep, none))

# HTF bias really is the finished-candle trend
h = resample_ohlc(data, "1h")
h["trend"] = structure(h, n=1).trend
trend = align_htf(data, h, "1h", ["trend"]).htf_trend.to_numpy()
out = SmcV1(**{**cfg, "killzones": "", "htf": "1h"}).generate_signals(data)
longs = out.entry.to_numpy() == 1
shorts = out.entry.to_numpy() == -1
check("longs only with a bullish HTF trend, shorts only bearish",
      (trend[longs] == 1).all() and (trend[shorts] == -1).all())

# ---- the look-ahead guard, with every filter switched on --------------------------------
sample = data.iloc[:6000]
for label, params in [
    ("default", {}),
    ("all filters on", {"require_sweep": True, "htf": "1h", "sweep_levels": "asia,pd,swing"}),
    ("body zones, choch only", {"zone": "body", "kinds": "choch", "htf": "none"}),
]:
    problems = check_no_lookahead(lambda p=params: SmcV1(**{**cfg, **p}), sample)
    check(f"no look-ahead: {label}", problems == [])

print("\nAll SMC v1 tests passed.")
