"""Tests for strategies/smc_limit.py.

Usage:
    python -m src.test_smc_limit
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtester import Backtester
from src.execution import ExecutionModel
from src.lookahead import check_no_lookahead

spec = importlib.util.spec_from_file_location("smc_limit", Path("strategies/smc_limit.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SmcLimit = module.SmcLimit


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


# Same hand-built chart as test_smc: BOS at 9, order block 97-99 (created on
# candle 9), price returns to it on candle 13.
CHART = [
    (100, 101, 99, 100.5), (100.5, 102, 100, 101.5), (101.5, 105, 101, 104),
    (104, 104.5, 101.5, 102), (102, 102.5, 100, 100.5), (100.5, 101, 98, 98.5),
    (98.5, 99, 97, 97.5), (97.5, 101, 97.4, 100.5), (100.5, 104, 100.4, 103.5),
    (103.5, 106.5, 103, 106), (106, 107, 105, 106.5), (106.5, 106.8, 103, 104),
    (104, 104.5, 100, 100.5), (100.5, 100.9, 98.5, 99.5),
]
EXTRA = [(99.6, 100, 99, 99.8), (99.8, 109.5, 99.5, 109), (109, 110, 108, 109)]
BASE = dict(swing=2, lookback=3, buffer=0.5, min_risk=1.0, htf="none")
NY_AM = "2025-01-07 12:00"     # 07:00 New York
NIGHT = "2025-01-07 03:00"     # 22:00 New York
EARLY = "2025-01-07 11:00"     # 06:00 New York: killzone opens at candle 12


def signals(rows, start, **params):
    s = SmcLimit(**{**BASE, **params})
    return s, s.generate_signals(frame(rows, start))


# ---- the hand-built setup ----------------------------------------------------------
s, out = signals(CHART, NY_AM)
rows = list(out.index[out.entry != 0])
check("a limit order is re-posted every candle while the zone is alive", rows == [9, 10, 11, 12])
check("price is the zone edge nearest the market, stop beyond the zone",
      (out.entry_price[rows] == 99.0).all() and (out.stop_price[rows] == 96.5).all())
check("orders last one candle and carry the 3R target",
      (out.valid_for[rows] == 1).all() and (out.target_r[rows] == 3.0).all())
check("no order once the zone was used (retest on candle 13)", out.entry[13] == 0)

d = frame(CHART + EXTRA, NY_AM)
bt = Backtester(d, SmcLimit(**BASE))
trades = bt.simulate()
check("one long trade", len(trades) == 1 and trades[0].side == "long")
check("filled AT THE ZONE on the retest candle, not at the next open",
      trades[0].entry_price == 99.0 and trades[0].entry_time == d.timestamp[13])
check("exact 3R target from the zone price (99 + 3 x 2.5)",
      trades[0].exit_reason == "take_profit" and abs(trades[0].pnl - 7.5) < 1e-9)

# ---- filters ---------------------------------------------------------------------------
s, out = signals(CHART, EARLY)
check("killzone is tested on the candle the order will be LIVE in",
      list(out.index[out.entry != 0]) == [11, 12])
s, out = signals(CHART, NIGHT)
funnel = list(s.diagnostics.values())
check("outside the killzone: no orders, funnel shows where they died",
      (out.entry == 0).all() and funnel[0] > 0 and funnel[1] == 0)
s, out = signals(CHART, NIGHT, killzones="")
check("killzone off: orders return", (out.entry != 0).any())
s, out = signals(CHART, NY_AM, allow_long=False)
check("longs can be disabled", (out.entry == 0).all())
s, out = signals(CHART, NY_AM, min_risk=5.0)
check("risk limits remove tiny stops", (out.entry == 0).all())
s, out = signals(CHART, NY_AM, entry_at=0.5)
check("entry_at=0.5 rests in the middle of the zone", (out.entry_price[out.entry != 0] == 98.0).all())
s, out = signals(CHART, NY_AM, require_sweep=True, sweep_levels="swing")
check("sweep required but none happened", (out.entry == 0).all())

for label, params in [("source", {"source": "xyz"}), ("entry_at", {"entry_at": 1.5}),
                      ("killzone", {"killzones": "tokyo"}), ("htf", {"htf": "7m"})]:
    try:
        SmcLimit(**params); ok = False
    except ValueError:
        ok = True
    check(f"rejects bad {label}", ok)

# ---- random markets --------------------------------------------------------------------
def week_data(start, weeks, seed):
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=weeks * 7 * 288, freq="5min", tz="UTC")
    wd, hr = times.weekday, times.hour
    times = times[~(((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 22)) | (hr == 21))]
    n = len(times)
    close = 2000 + rng.normal(0, 0.8, n).cumsum()
    o = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "timestamp": times, "open": o,
        "high": np.maximum(o, close) + rng.uniform(0, .5, n),
        "low": np.minimum(o, close) - rng.uniform(0, .5, n),
        "close": close, "volume": 100})


data = week_data("2021-01-04", 110, 17)
cfg = dict(swing=2, max_risk=40.0, killzones="", htf="none")
n_orders = int((SmcLimit(**cfg).generate_signals(data).entry != 0).sum())
check("random data produces orders", n_orders > 500)

free = Backtester(data, SmcLimit(**cfg), execution=ExecutionModel(0.0, 0.0))
t_free = free.simulate()
wins = sum(t.pnl > 0 for t in t_free)
rate = wins / len(t_free) * 100
se = (0.25 * 0.75 / len(t_free)) ** 0.5 * 100
print(f"      info: {len(t_free)} trades, win rate {rate:.1f}% (fair 1:3 bet = 25.0%, std error {se:.1f})")
check("on random prices with no costs a fair 1:3 bet wins about 25% (no free money from the fill model)",
      abs(rate - 25.0) < 3.5 * se and len(t_free) > 200)
costly = Backtester(data, SmcLimit(**cfg), execution=ExecutionModel(0.30, 0.05)).simulate()
check("realistic costs reduce the result", sum(t.pnl for t in costly) < sum(t.pnl for t in t_free))

# ---- the look-ahead guard, with filters on ------------------------------------------------
sample = data.iloc[:6000]
for label, params in [("defaults", {}), ("fvg source", {"source": "fvg"}),
                      ("all filters", {"require_sweep": True, "htf": "1h"})]:
    problems = check_no_lookahead(lambda p=params: SmcLimit(**{**cfg, **p}), sample)
    check(f"no look-ahead: {label}", problems == [])

print("\nAll SMC limit tests passed.")
