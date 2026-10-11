"""Tests for the diagnosis tool and the engine's order accounting.

Usage:
    python -m src.test_diagnose
"""

import time

import numpy as np
import pandas as pd

from src.backtester import Backtester
from src.diagnose import diagnose
from src.execution import ExecutionModel
from src.timeframes import max_entry_gap_for


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


rng = np.random.default_rng(12)
times = pd.date_range("2024-01-01 22:00", periods=9 * 7 * 288, freq="5min", tz="UTC")
wd, hr = times.weekday, times.hour
times = times[~(((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 22)) | (hr == 21))]
n = len(times)
close = 2000 + rng.normal(0, 0.7, n).cumsum()
o = np.concatenate([[close[0]], close[:-1]])
data = pd.DataFrame({"timestamp": times, "open": o,
                     "high": np.maximum(o, close) + 0.3, "low": np.minimum(o, close) - 0.3,
                     "close": close, "volume": 1})
execution, gap = ExecutionModel(0.30, 0.05), max_entry_gap_for("5m")
SIZES = (2000, 4000, 8000)
every = lambda d, k: (np.arange(len(d)) % k) == 0


def run(strategy, **kw):
    return diagnose(data, lambda: strategy, execution, gap, sizes=SIZES, call_timeout=kw.get("t", 30))


def has(report, text):
    return any(text in r for r in report["reasons"])


class Never:
    def generate_signals(self, d):
        return pd.DataFrame({"signal": np.zeros(len(d), dtype=int)}, index=d.index)


class WrongStops:
    def generate_signals(self, d):
        c = d["close"].to_numpy()
        return pd.DataFrame({"entry": np.where(every(d, 40), 1, 0), "stop_price": c + 1.0,
                             "target_r": 2.0}, index=d.index)


class FarLimits:
    def generate_signals(self, d):
        c = d["close"].to_numpy()
        e = np.where(every(d, 20), 1, 0)
        return pd.DataFrame({"entry": e, "entry_price": c - 30, "stop_price": c - 31,
                             "target_r": 2.0, "valid_for": 1.0}, index=d.index)


class OverTrader:
    def generate_signals(self, d):
        return pd.DataFrame({"entry": np.ones(len(d), dtype=int), "stop_distance": 20.0,
                             "target_r": 30.0}, index=d.index)


class RepostNear:
    """A setup lasting 30 candles, re-posted each candle just below the market."""
    def generate_signals(self, d):
        c = d["close"].to_numpy()
        live = (np.arange(len(d)) % 100) < 30
        return pd.DataFrame({"entry": np.where(live, 1, 0), "entry_price": c - 0.2,
                             "stop_price": c - 3.0, "target_r": 2.0, "valid_for": 1.0}, index=d.index)


class RepostFar(RepostNear):
    def generate_signals(self, d):
        frame = super().generate_signals(d)
        c = d["close"].to_numpy()
        frame["entry_price"] = c - 30
        frame["stop_price"] = c - 33
        return frame


class TinyStops:
    def generate_signals(self, d):
        return pd.DataFrame({"entry": np.where(every(d, 10), 1, 0), "stop_distance": 0.3,
                             "target_r": 2.0}, index=d.index)


class Healthy:
    def generate_signals(self, d):
        return pd.DataFrame({"entry": np.where(every(d, 25), 1, 0), "stop_distance": 3.0,
                             "target_r": 2.0}, index=d.index)


class Quadratic:
    def generate_signals(self, d):
        total = 0
        for i in range(len(d)):
            for _ in range(i):
                pass
        return pd.DataFrame({"signal": np.zeros(len(d), dtype=int)}, index=d.index)


class Linear:
    def generate_signals(self, d):
        c = d["close"]
        return pd.DataFrame({"signal": np.where(c > c.rolling(50).mean(), 1, -1)}, index=d.index)


class Hanging:
    def generate_signals(self, d):
        time.sleep(30)


class Garbage:
    def generate_signals(self, d):
        return {"a": 1}


check("a strategy that never signals is named as such", has(run(Never()), "NEVER SIGNALS"))
r = run(WrongStops())
check("wrong-side stops are named, with counts", has(r, "REJECTED AS INVALID") and r["fate"]["stats"]["invalid"] > 100)
r = run(FarLimits())
check("resting orders that never fill are named, with the distance",
      has(r, "NEVER REACH THEIR PRICE") and "median $30.00" in " ".join(r["reasons"]))
r = run(OverTrader())
r = run(RepostNear())
check("orders re-posted each candle that DO fill are not flagged as unfilled "
      f"({r['output']['episodes']} setups, {r['fate']['stats']['filled']} filled)",
      not has(r, "NEVER REACH THEIR PRICE") and r["output"]["repost_share"] > 0.9)
r = run(RepostFar())
check("re-posted orders that never fill are judged per SETUP, not per order",
      has(r, "NEVER REACH THEIR PRICE") and "separate setups" in " ".join(r["reasons"]))
r = run(OverTrader())
check("an over-trader whose entries are dropped is named", has(r, "TRADE WAS ALREADY OPEN") and r["fate"]["stats"]["dropped_busy"] > 1000)
check("stops that are tiny against costs are named", has(run(TinyStops()), "STOPS ARE SMALL AGAINST COSTS"))
check("a healthy strategy gets a clean bill of health", run(Healthy())["reasons"] == [])

slow = run(Quadratic())
check(f"a quadratic strategy is flagged as slow (exponent {slow['speed']['exponent']:.2f})",
      slow["speed"]["exponent"] > 1.5 and has(slow, "SLOW"))
fast = run(Linear())
check("a linear strategy is not flagged", not has(fast, "SLOW") and not has(fast, "VERY SLOW"))
hung = run(Hanging(), t=1)
check("a strategy that hangs is reported without freezing the tool", has(hung, "VERY SLOW"))
bad = run(Garbage())
check("unusable output is explained", bad["error"] and "cannot be used" in bad["error"])

# ---- the engine's accounting must add up --------------------------------------------------
for label, strategy in [("far limits", FarLimits()), ("over-trader", OverTrader()),
                        ("wrong stops", WrongStops()), ("healthy", Healthy())]:
    bt = Backtester(data, strategy, execution=execution, max_entry_gap=gap)
    bt.simulate()
    s = bt.order_stats
    accounted = (s["filled"] + s["expired_unfilled"] + s["cancelled"] + s["dropped_busy"]
                 + s["invalid"] + s["gap_blocked"] + s["overflow"])
    check(f"order accounting adds up ({label}: posted {s['posted']}, accounted {accounted}, pending at end <= 2)",
          0 <= s["posted"] - accounted <= 2)

print("\nAll diagnose tests passed.")
