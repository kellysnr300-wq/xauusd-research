"""Tests for the placebo test.

Usage:
    python -m src.test_placebo
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.execution import ExecutionModel
from src.placebo import (
    NEW_YORK, build_report, rotate_state, run_placebo, transplant_events,
)
from src.timeframes import max_entry_gap_for, resample_ohlc


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def market(weeks, seed, start="2024-01-01 22:00", drift=0.0):
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=weeks * 7 * 288, freq="5min", tz="UTC")
    wd, hr = times.weekday, times.hour
    times = times[~(((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 22)) | (hr == 21))]
    n = len(times)
    close = 2000 + (drift + rng.normal(0, 0.7, n)).cumsum()
    o = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "timestamp": times, "open": o,
        "high": np.maximum(o, close) + rng.uniform(0, .4, n),
        "low": np.minimum(o, close) - rng.uniform(0, .4, n),
        "close": close, "volume": 100})


# ======================================================================
# 1. Transplanting orders keeps geometry, clock time and block shape
# ======================================================================
data = market(70, 1)                       # crosses both daylight-saving changes
n = len(data)
close = data["close"].to_numpy()
ts = pd.DatetimeIndex(data["timestamp"])
local = ts.tz_convert(NEW_YORK)
rng = np.random.default_rng(0)

starts = np.sort(rng.choice(np.arange(500, n - 500), 120, replace=False))
rows = sorted({int(s + k) for s in starts for k in range(int(rng.integers(1, 5)))})
entry = np.zeros(n, int); price = np.full(n, np.nan); stop = np.full(n, np.nan)
r = np.full(n, np.nan); ident = np.full(n, np.nan)
for number, row in enumerate(rows, start=1):
    side = 1 if row % 2 else -1
    entry[row] = side
    price[row] = close[row] - side * (0.4 + 0.01 * number)
    stop[row] = price[row] - side * (1.0 + 0.003 * number)
    r[row] = 2.0 + 0.001 * number
    ident[row] = number
signals = pd.DataFrame({"entry": entry, "entry_price": price, "stop_price": stop,
                        "target_r": r, "valid_for": ident}, index=data.index)

new, moved, total = transplant_events(signals, data, np.random.default_rng(5), max_shift_days=60)
new_rows = np.flatnonzero(new["entry"].to_numpy() != 0)
check(f"most orders land on a real candle ({moved}/{total})", moved / total > 0.85)

origin = {int(ident[row]): row for row in rows}
bad_geometry = bad_clock = bad_shift = crossed_dst = 0
for row in new_rows:
    k = int(new["valid_for"].iloc[row])
    src = origin[k]
    same = (
        new["entry"].iloc[row] == entry[src]
        and np.isclose(new["entry_price"].iloc[row] - close[row], price[src] - close[src])
        and np.isclose(new["stop_price"].iloc[row] - close[row], stop[src] - close[src])
        and np.isclose(new["target_r"].iloc[row], r[src])
    )
    bad_geometry += not same
    a, b = local[src], local[row]
    bad_clock += (a.hour, a.minute) != (b.hour, b.minute)
    days = abs((b.normalize().tz_localize(None) - a.normalize().tz_localize(None)).days)
    bad_shift += not (1 <= days <= 60)
    crossed_dst += (ts[src].hour != ts[row].hour) and (a.hour, a.minute) == (b.hour, b.minute)
check("side, distance from market, stop size and R are preserved exactly", bad_geometry == 0)
check("every order keeps its New York clock time", bad_clock == 0)
check("every order moves 1 to 60 days, never 0", bad_shift == 0)
check(f"daylight saving handled: {crossed_dst} orders moved across a clock change keep NY time but shift UTC hour",
      crossed_dst > 0)

consecutive = [(a, b) for a, b in zip(rows[:-1], rows[1:]) if b == a + 1]
destination = {int(new["valid_for"].iloc[row]): row for row in new_rows}
kept = [(destination[int(ident[a])], destination[int(ident[b])])
        for a, b in consecutive if int(ident[a]) in destination and int(ident[b]) in destination]
check("blocks of consecutive orders stay consecutive", len(kept) > 20 and all(y == x + 1 for x, y in kept))

again, *_ = transplant_events(signals, data, np.random.default_rng(5), max_shift_days=60)
other, *_ = transplant_events(signals, data, np.random.default_rng(6), max_shift_days=60)
check("same seed gives the same placebo, another seed a different one",
      again.equals(new) and not other.equals(new))

# ======================================================================
# 2. Rotating a position series
# ======================================================================
state = pd.DataFrame({"signal": np.repeat([0, 1, 0, -1, 1, 0, -1], 40)[:260]})
frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=260, freq="1D", tz="UTC"),
                      "open": 1., "high": 2., "low": .5, "close": 1.5})
rotated, *_ = rotate_state(state, frame, np.random.default_rng(3))
check("rotation keeps exactly the same number of long, short and flat candles",
      sorted(np.bincount(rotated["signal"] + 1)) == sorted(np.bincount(state["signal"] + 1)))
check("rotation actually moves the series", not rotated["signal"].equals(state["signal"]))

# ======================================================================
# 3. Verdict logic
# ======================================================================
def fake(total, trades=100, wins=30):
    return {"trade_count": trades, "wins": wins, "win_rate": wins / trades * 100,
            "total_pnl": total, "profit_factor": None}

crowd = [fake(t) for t in np.linspace(-100, 100, 40)]
check("real far above the crowd: BETTER", build_report(fake(300), crowd, "event")["verdict"] == "BETTER than random timing")
check("real far below the crowd: WORSE", build_report(fake(-300), crowd, "event")["verdict"] == "WORSE than random timing")
check("real in the middle: INDISTINGUISHABLE", build_report(fake(0), crowd, "event")["verdict"] == "INDISTINGUISHABLE from random timing")
check("few trades: refuses to judge", build_report(fake(300, trades=10), crowd, "event")["verdict"] == "too few trades to judge")
check("p-value is never zero", build_report(fake(1e9), crowd, "event")["p_better"] > 0)

# ======================================================================
# 4. The test must not cry wolf, and must catch real skill
# ======================================================================
small = market(5, 11)
execution, gap = ExecutionModel(0.0, 0.0), max_entry_gap_for("5m")


class RandomTimer:
    """No skill: market entries at random moments with fixed geometry."""
    def __init__(self, seed):
        self.seed = seed

    def generate_signals(self, d):
        rng = np.random.default_rng(self.seed)
        entry = np.where(rng.random(len(d)) < 0.04, rng.choice([-1, 1], len(d)), 0)
        return pd.DataFrame({"entry": entry, "stop_distance": 1.0, "target_r": 2.0}, index=d.index)


p_values = []
for seed in range(40):
    rep = run_placebo(small, lambda s=seed: RandomTimer(s), runs=19, seed=seed,
                      execution=execution, max_gap=gap, max_shift_days=10)
    p_values.append(rep["p_better"])
p_values = np.array(p_values)
check(f"no skill: p-values spread out, mean {p_values.mean():.2f} (should be about 0.5)",
      0.33 <= p_values.mean() <= 0.67)
check(f"no skill: only {np.mean(p_values <= 0.05):.0%} of 40 skill-free strategies flagged (should be about 5%)",
      np.mean(p_values <= 0.05) <= 0.15)


class Peeker:
    """Cheats on purpose (looks 6 candles ahead). Only for this test."""
    def generate_signals(self, d):
        c = d["close"].to_numpy()
        future = np.concatenate([c[6:], np.full(6, np.nan)]) - c
        entry = np.where(future > 1.5, 1, 0)
        entry[::7] = entry[::7]            # keep it sparse enough to be tradeable
        mask = (np.arange(len(d)) % 7) == 0
        entry = np.where(mask, entry, 0)
        return pd.DataFrame({"entry": entry, "stop_distance": 1.0, "target_r": 1.0}, index=d.index)


rep = run_placebo(small, Peeker, runs=19, seed=3, execution=execution, max_gap=gap, max_shift_days=10)
check(f"real skill (a deliberate look-ahead): detected, p = {rep['p_better']:.3f}, z = {rep['z_score']:.1f}",
      rep["verdict"] == "BETTER than random timing" and rep["p_better"] <= 0.05 + 1e-9)

# ======================================================================
# 5. State strategies run end to end
# ======================================================================
spec = importlib.util.spec_from_file_location("gold_trend", Path("strategies/gold_trend.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
daily = resample_ohlc(market(300, 2, drift=0.004), "1d")
rep = run_placebo(daily, module.GoldTrend, runs=12, seed=1, execution=execution,
                  max_gap=max_entry_gap_for("1d"))
check("state strategy: report is complete",
      rep["mode"] == "state" and rep["valid_runs"] >= 10 and "p_better" in rep and rep["real"]["trade_count"] > 5)

print("\nAll placebo tests passed.")
