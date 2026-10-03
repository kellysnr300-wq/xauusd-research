"""Scenario tests for the Backtester event mode.

Usage:
    python -m src.test_event_mode
"""

import pandas as pd

from src.backtester import Backtester
from src.execution import ExecutionModel
from src.strategies import Strategy


class Scripted(Strategy):
    def __init__(self, entry, stop, r=3.0):
        self.entry, self.stop, self.r = entry, stop, r

    def generate_signals(self, data):
        out = pd.DataFrame(index=data.index)
        out["entry"] = self.entry
        out["stop_price"] = self.stop
        out["target_r"] = [self.r if e else float("nan") for e in self.entry]
        return out


def make_data(rows):
    times = pd.date_range("2025-01-06", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": times,
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
    })


def run(rows, entry, stop, r=3.0, execution=None, data=None):
    bt = Backtester(data if data is not None else make_data(rows),
                    Scripted(entry, stop, r), execution=execution)
    return bt, bt.simulate()


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def approx(a, b):
    return abs(a - b) < 1e-9


N = float("nan")

# 1. Long: entry at next open 100, stop 98 (risk 2) -> target 106 -> +6.
rows = [(100, 101, 99, 100), (100, 101, 99, 100.5), (100.5, 107, 100, 106),
        (106, 107, 105, 106)]
bt, t = run(rows, [1, 0, 0, 0], [98, N, N, N])
check("one trade", len(t) == 1)
check("take profit at exactly 3R", t[0].exit_reason == "take_profit")
check("pnl = 3 x risk", approx(t[0].pnl, 6.0))
check("entry at next open", t[0].entry_price == 100)

# 2. Stop hit -> loses exactly 1R.
rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 100.5, 97, 97.5),
        (97.5, 98, 97, 97.5)]
bt, t = run(rows, [1, 0, 0, 0], [98, N, N, N])
check("stop loss", t[0].exit_reason == "stop_loss")
check("loss = 1R", approx(t[0].pnl, -2.0))

# 3. Short mirrored: entry 100, stop 102 (risk 2) -> target 94.
rows = [(100, 101, 99, 100), (100, 100.5, 99, 99.5), (99.5, 100, 93, 94),
        (94, 95, 93, 94)]
bt, t = run(rows, [-1, 0, 0, 0], [102, N, N, N])
check("short take profit", t[0].exit_reason == "take_profit"
      and approx(t[0].pnl, 6.0))

# 4. Entry events are ignored while a trade is open.
rows = [(100, 101, 99, 100)] * 3 + [(100, 107, 99, 106)] + [(106, 107, 105, 106)]
bt, t = run(rows, [1, 1, 1, 0, 0], [98, 98, 98, N, N])
check("no pyramiding", len(t) == 1)

# 5. New entry allowed after the first trade ends.
rows = [(100, 101, 99, 100), (100, 107, 99, 106), (106, 107, 105, 106),
        (106, 107, 105, 106), (106, 120, 105, 118), (118, 119, 117, 118)]
bt, t = run(rows, [1, 0, 1, 0, 0, 0], [98, N, 104, N, N, N])
check("second trade after exit", len(t) == 2)

# 6. Stop on the wrong side of the open -> skipped, counted.
rows = [(100, 101, 99, 100)] * 4
bt, t = run(rows, [1, 0, 0, 0], [101, N, N, N])
check("invalid stop skipped", len(t) == 0 and bt.skipped_entries == 1)

# 7. Stop and target both inside one candle -> stop wins.
rows = [(100, 101, 99, 100), (100, 110, 95, 100), (100, 101, 99, 100)]
bt, t = run(rows, [1, 0, 0], [98, N, N])
check("stop wins when both hit", t[0].exit_reason == "stop_loss")

# 8. Gap through the stop fills at the open.
rows = [(100, 101, 99, 100), (100, 101, 99, 100), (95, 96, 94, 95),
        (95, 96, 94, 95)]
bt, t = run(rows, [1, 0, 0, 0], [98, N, N, N])
check("gap fills at open", approx(t[0].exit_price, 95))

# 9. Costs are applied on both sides.
rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 107, 99, 106),
        (106, 107, 105, 106)]
bt, t = run(rows, [1, 0, 0, 0], [98, N, N, N],
            execution=ExecutionModel(1.0, 0.0))
check("costs applied", approx(t[0].entry_price, 100.5)
      and approx(t[0].exit_price, 105.5))

# 10. Entry skipped across a big time gap.
data = make_data([(100, 101, 99, 100)] * 4)
data.loc[1:, "timestamp"] += pd.Timedelta(hours=48)
bt, t = run(None, [1, 0, 0, 0], [98, N, N, N], data=data)
check("skipped across gap", len(t) == 0)

# 11. Open trade closed at end of data.
rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 102, 99, 101)]
bt, t = run(rows, [1, 0, 0], [98, N, N])
check("end of data close", t[0].exit_reason == "end_of_data")

# 12. Existing 'signal' mode still works next to event mode.
class Plain(Strategy):
    def generate_signals(self, data):
        out = pd.DataFrame(index=data.index)
        out["signal"] = [1, 1, 0, 0]
        return out
bt = Backtester(make_data([(100, 101, 99, 100)] * 4), Plain())
check("signal mode unaffected", len(bt.simulate()) == 1)

print("\nAll event-mode tests passed.")
