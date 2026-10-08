"""Tests for the sweep tool's bookkeeping.

Usage:
    python -m src.test_sweep
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.execution import ExecutionModel
from src.sweep import buy_and_hold, evaluate, parse_grid, variants
from src.timeframes import max_entry_gap_for, resample_ohlc

spec = importlib.util.spec_from_file_location("gold_trend", Path("strategies/gold_trend.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GoldTrend = module.GoldTrend


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def market(seed, years=7, drift=0.012):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2017-01-01 22:00", periods=years * 52 * 7 * 288, freq="5min", tz="UTC")
    wd, hr = times.weekday, times.hour
    times = times[~(((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 22)) | (hr == 21))]
    n = len(times)
    regime = np.repeat(rng.choice([-1.0, 1.0], size=n // 20000 + 1), 20000)[:n] * drift
    close = 2000 + (regime + rng.normal(0, 0.7, n)).cumsum()
    o = np.concatenate([[close[0]], close[:-1]])
    return resample_ohlc(pd.DataFrame({
        "timestamp": times, "open": o, "high": np.maximum(o, close) + .3,
        "low": np.minimum(o, close) - .3, "close": close, "volume": 1}), "1d")


data = market(11)
exe, gap = ExecutionModel(0.30, 0.05), max_entry_gap_for("1d")
r = evaluate(data, lambda: GoldTrend(), 2020, exe, gap)

check("the run produced trades", r["trades"] >= 10)
check("dev + check = total (every trade is in exactly one period)",
      abs(r["dev"] + r["check"] - r["total"]) < 1e-6)
check("long + short = total", abs(r["long"] + r["short"] - r["total"]) < 1e-6)
check("yearly P&L adds up to the total", abs(sum(r["yearly"].values()) - r["total"]) < 1e-6)
check("drawdown is never negative", r["max_drawdown"] >= 0)

early = evaluate(data, lambda: GoldTrend(), 2018, exe, gap)
check("moving the split moves P&L between periods but not the total",
      abs(early["total"] - r["total"]) < 1e-6 and early["dev"] != r["dev"])
check("same inputs give identical results (deterministic)",
      evaluate(data, lambda: GoldTrend(), 2020, exe, gap)["total"] == r["total"])

check("percent: dev + check = total", abs(r["dev_pct"] + r["check_pct"] - r["total_pct"]) < 1e-6)
check("percent: long + short = total", abs(r["long_pct"] + r["short_pct"] - r["total_pct"]) < 1e-6)
check("percent: yearly adds up", abs(sum(r["yearly_pct"].values()) - r["total_pct"]) < 1e-6)
check("per-period side split adds up to the total",
      abs(r["long_dev"] + r["short_dev"] + r["long_check"] + r["short_check"] - r["total"]) < 1e-6)
check("per-period side split matches the period totals",
      abs(r["long_dev"] + r["short_dev"] - r["dev"]) < 1e-6
      and abs(r["long_check"] + r["short_check"] - r["check"]) < 1e-6)
best = max(r["yearly_pct"].values())
check("'without the best year' removes exactly that year",
      abs(r["ex_best_year_pct"] - (r["total_pct"] - best)) < 1e-6
      and r["yearly_pct"][r["best_year"]] == best)

hold = buy_and_hold(data, 2020)
check("buy and hold percent is the compounded change",
      abs(hold["total_pct"] - (data.close.iloc[-1] / data.open.iloc[0] - 1) * 100) < 1e-9)
check("buy and hold: periods add up", abs(hold["dev"] + hold["check"] - hold["total"]) < data.close.max())
check("buy and hold total is last close minus first open",
      abs(hold["total"] - (data.close.iloc[-1] - data.open.iloc[0])) < 1e-6)

grid = parse_grid(["entry_n=40,55,80", "exit_n=15,20", "allow_short=true,false", "label=abc"])
check("grid parsing: numbers, booleans and text",
      grid == {"entry_n": [40, 55, 80], "exit_n": [15, 20], "allow_short": [True, False], "label": ["abc"]})
base = {"entry_n": 55, "exit_n": 20}
combos = list(variants(base, {"entry_n": [40, 55], "exit_n": [15, 20]}))
check("base is first and duplicates of the base are dropped",
      combos[0][0] == "base" and len(combos) == 4)

try:
    parse_grid(["broken"]); ok = False
except ValueError:
    ok = True
check("a malformed grid is rejected", ok)

print("\nAll sweep tests passed.")
