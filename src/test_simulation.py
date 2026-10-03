"""Deterministic scenario tests for Backtester.simulate().

Usage:
    python -m src.test_simulation
"""

import pandas as pd

from src.backtester import Backtester
from src.execution import ExecutionModel
from src.strategies import Strategy


class ScriptedStrategy(Strategy):
    """Returns pre-written signals (and optional stop/target distances)."""

    def __init__(self, signals, stop=None, target=None):
        self.signals = signals
        self.stop = stop
        self.target = target

    def generate_signals(self, data):
        out = pd.DataFrame(index=data.index)
        out["signal"] = self.signals
        if self.stop is not None:
            out["stop_distance"] = self.stop
        if self.target is not None:
            out["target_distance"] = self.target
        return out


def make_data(rows, step_minutes=5):
    """rows = list of (open, high, low, close)."""
    times = pd.date_range(
        "2025-01-06 00:00", periods=len(rows),
        freq=f"{step_minutes}min", tz="UTC",
    )
    return pd.DataFrame({
        "timestamp": times,
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
    })


def run(rows, signals, stop=None, target=None, execution=None, **kw):
    bt = Backtester(
        make_data(rows),
        ScriptedStrategy(signals, stop, target),
        execution=execution,
        **kw,
    )
    return bt.simulate()


def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    if not condition:
        raise SystemExit(1)


def approx(a, b):
    return abs(a - b) < 1e-9


# 1. Signal on bar 0 enters at the OPEN of bar 1; exit signal at bar 2
#    exits at the open of bar 3.
rows = [
    (100, 101, 99, 100),
    (102, 103, 101, 102),
    (104, 105, 103, 104),
    (106, 107, 105, 106),
    (108, 109, 107, 108),
]
t = run(rows, [1, 1, 0, 0, 0])
check("one trade", len(t) == 1)
check("entry at next open", t[0].entry_price == 102)
check("exit at next open", t[0].exit_price == 106)
check("pnl", approx(t[0].pnl, 4))
check("exit reason signal", t[0].exit_reason == "signal")

# 2. Costs: spread 1, slippage 0.2 -> entry +0.7, exit -0.7.
t = run(rows, [1, 1, 0, 0, 0], execution=ExecutionModel(1.0, 0.2))
check("entry includes costs", approx(t[0].entry_price, 102.7))
check("exit includes costs", approx(t[0].exit_price, 105.3))

# 3. Short trade.
t = run(rows, [-1, -1, 0, 0, 0])
check("short pnl negative in uptrend", approx(t[0].pnl, -4))

# 4. Stop-loss hit inside entry candle's range.
rows = [
    (100, 101, 99, 100),
    (100, 101, 94, 95),   # entry 100, stop 95 hit (low 94)
    (95, 96, 94, 95),
]
t = run(rows, [1, 0, 0], stop=[5, 5, 5])
check("stop loss exit", t[0].exit_reason == "stop_loss")
check("stop fill price", approx(t[0].exit_price, 95))
check("stop pnl", approx(t[0].pnl, -5))

# 5. Take-profit.
rows = [
    (100, 101, 99, 100),
    (100, 111, 99, 105),
    (105, 106, 104, 105),
]
t = run(rows, [1, 0, 0], target=[10, 10, 10])
check("take profit exit", t[0].exit_reason == "take_profit")
check("take profit pnl", approx(t[0].pnl, 10))

# 6. Stop and target both touched in one candle -> stop wins.
rows = [
    (100, 101, 99, 100),
    (100, 115, 90, 100),
    (100, 101, 99, 100),
]
t = run(rows, [1, 0, 0], stop=[5, 5, 5], target=[10, 10, 10])
check("stop wins when both hit", t[0].exit_reason == "stop_loss")

# 7. Gap through the stop fills at the open, not the stop.
rows = [
    (100, 101, 99, 100),
    (100, 101, 99, 100),   # entry 100, stop 95
    (90, 91, 89, 90),      # gaps open below stop
    (90, 91, 89, 90),
]
t = run(rows, [1, 1, 1, 1], stop=[5, 5, 5, 5])
check("gap fill at open", approx(t[0].exit_price, 90))
check("gap loss larger than stop", approx(t[0].pnl, -10))

# 8. No instant re-entry after a stop while the signal persists.
check("no re-entry after stop", len(t) == 1)

# 9. Re-entry allowed once the signal resets.
rows = [(100, 101, 99, 100)] * 2 + [(100, 101, 94, 95)] + \
       [(95, 96, 94, 95)] * 4
t = run(rows, [1, 1, 1, 0, 1, 1, 1], stop=[5] * 7)
check("re-entry after signal reset", len(t) == 2)

# 10. Position open at the end is closed at the last close.
rows = [
    (100, 101, 99, 100),
    (100, 101, 99, 100),
    (100, 103, 99, 102),
]
t = run(rows, [1, 1, 1])
check("end of data close", t[0].exit_reason == "end_of_data")
check("end of data price", t[0].exit_price == 102)

# 11. Signal on the final candle is ignored (no next candle).
t = run(rows, [0, 0, 1])
check("last-candle signal ignored", len(t) == 0)

# 12. Entry skipped when the next candle is after a big gap.
data = make_data([(100, 101, 99, 100)] * 4)
data.loc[1:, "timestamp"] += pd.Timedelta(hours=48)
bt = Backtester(data, ScriptedStrategy([1, 0, 0, 0]))
check("entry skipped across gap", len(bt.simulate()) == 0)

# 13. simulate() can be called twice with identical results.
bt = Backtester(make_data(rows), ScriptedStrategy([1, 1, 0]))
first = len(bt.simulate())
second = len(bt.simulate())
check("repeatable", first == second == len(bt.trades))

# 14. Portfolio equity matches summed trade pnl.
rows = [
    (100, 101, 99, 100),
    (102, 103, 101, 102),
    (104, 105, 103, 104),
    (106, 107, 105, 106),
    (108, 109, 107, 108),
]
bt = Backtester(make_data(rows), ScriptedStrategy([1, 1, 0, 0, 0]))
bt.simulate()
check("portfolio equity", approx(bt.portfolio.equity, 10_000 + 4))

print("\nAll simulation tests passed.")
