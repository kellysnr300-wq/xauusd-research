"""Tests for intrabar resolution and the accuracy report.

Usage:
    python -m src.test_accuracy
"""

import tempfile
from pathlib import Path

import pandas as pd

from src.accuracy import accuracy_report, wilson_interval
from src.backtester import Backtester
from src.execution import ExecutionModel
from src.intrabar import IntrabarResolver


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


class Scripted:
    def __init__(self, signal, stop, target):
        self.signal, self.stop, self.target = signal, stop, target

    def generate_signals(self, data):
        out = pd.DataFrame(index=data.index)
        out["signal"] = self.signal
        out["stop_distance"] = self.stop
        out["target_distance"] = self.target
        return out


def candles(rows, start="2025-01-07 09:55"):
    times = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": times,
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
    })


TMP = Path(tempfile.mkdtemp())


def write_minutes(day, rows):
    """rows: [(HH:MM, open, high, low, close)] in the real raw format."""
    lines = ["Etc/UTC,Open,High,Low,Close,Volume"]
    for hhmm, o, h, l, c in rows:
        lines.append(f"{day}T{hhmm}:00+00:00,{o},{h},{l},{c},1000")
    (TMP / f"{day}.csv").write_text("\n".join(lines) + "\n")


def resolver():
    return IntrabarResolver(path_for_day=lambda day: TMP / f"{day}.csv")


# Long trade: entry at 10:00 open 100, stop 97, target 106.
# The 10:00 candle touches both (low 96, high 107).
AMBIG = [(100, 101, 99, 100), (100, 107, 96, 105), (105, 106, 104, 105)]
sig = [1, 1, 0]


def run(rows, **kw):
    bt = Backtester(candles(rows), Scripted(sig, 3.0, 6.0), **kw)
    return bt, bt.simulate()


# ---- resolving with 1-minute data -------------------------------------------
write_minutes("2025-01-07", [
    ("10:00", 100, 100.5, 99.5, 100.2), ("10:01", 100.2, 107, 100.1, 106.5),
    ("10:02", 106.5, 106.6, 96, 97), ("10:03", 97, 98, 96.5, 97), ("10:04", 97, 98, 96.5, 97),
])
bt, t = run(AMBIG, resolver=resolver(), ambiguity="stop_first")
check("1-minute data: target came first", t[0].exit_reason == "take_profit")
check("trade is flagged ambiguous AND resolved", bt.trade_meta[0] == {"ambiguous": True, "resolved": True})

write_minutes("2025-01-07", [
    ("10:00", 100, 100.2, 96, 97), ("10:01", 97, 107, 96.5, 106), ("10:02", 106, 107, 105, 106),
    ("10:03", 106, 107, 105, 106), ("10:04", 106, 107, 105, 106),
])
bt, t = run(AMBIG, resolver=resolver(), ambiguity="target_first")
check("1-minute data: stop came first (even when policy favours target)", t[0].exit_reason == "stop_loss")

write_minutes("2025-01-07", [
    ("10:00", 100, 107, 96, 100), ("10:01", 100, 101, 99, 100), ("10:02", 100, 101, 99, 100),
    ("10:03", 100, 101, 99, 100), ("10:04", 100, 101, 99, 100),
])
bt, t = run(AMBIG, resolver=resolver(), ambiguity="stop_first")
check("both inside one minute: unresolved, policy stop_first", t[0].exit_reason == "stop_loss" and bt.trade_meta[0] == {"ambiguous": True, "resolved": False})
bt, t = run(AMBIG, resolver=resolver(), ambiguity="target_first")
check("both inside one minute: unresolved, policy target_first", t[0].exit_reason == "take_profit")

missing = IntrabarResolver(path_for_day=lambda day: TMP / "nope.csv")
bt, t = run(AMBIG, resolver=missing)
check("missing 1-minute file: falls back to policy and is counted",
      t[0].exit_reason == "stop_loss" and len(missing.missing_days) == 1)

bt, t = run(AMBIG)
check("no resolver: stop first (as before)", t[0].exit_reason == "stop_loss")
bt, t = run(AMBIG, ambiguity="target_first")
check("no resolver: target_first policy", t[0].exit_reason == "take_profit")

# ---- a candle that opens through a level has no ambiguity ------------------------
GAP_UP = [(100, 101, 99, 100), (100, 101, 99, 100.5), (106.5, 108, 96, 100)]
for policy in ("stop_first", "target_first"):
    bt, t = run(GAP_UP, ambiguity=policy)
    check(f"opens beyond the target -> target at the open ({policy})",
          t[0].exit_reason == "take_profit" and not bt.trade_meta[0]["ambiguous"])
GAP_DOWN = [(100, 101, 99, 100), (100, 101, 99, 100.5), (96.5, 107, 96, 100)]
for policy in ("stop_first", "target_first"):
    bt, t = run(GAP_DOWN, ambiguity=policy)
    check(f"opens beyond the stop -> stop at the open ({policy})",
          t[0].exit_reason == "stop_loss" and not bt.trade_meta[0]["ambiguous"])

# ---- ask-side triggers for shorts -----------------------------------------------
short = lambda rows, **kw: Backtester(
    candles(rows), Scripted([-1, -1, 0], 3.0, 6.0),
    execution=ExecutionModel(0.3, 0.0), **kw)
STOP_NEAR = [(100, 101, 99, 100), (100, 102.8, 99, 100), (100, 101, 99, 100)]
check("short stop at 103: bid high 102.8 does not trigger by default",
      len(short(STOP_NEAR).simulate()) == 1 and short(STOP_NEAR).simulate()[0].exit_reason == "end_of_data")
check("short stop triggers on the ask (102.8 + 0.3 >= 103)",
      short(STOP_NEAR, ask_triggers=True).simulate()[0].exit_reason == "stop_loss")
TP_NEAR = [(100, 101, 99, 100), (100, 100.5, 93.9, 100), (100, 101, 99, 100)]
check("short target 94: bid low 93.9 triggers by default",
      short(TP_NEAR).simulate()[0].exit_reason == "take_profit")
check("short target needs the ask to reach it (93.9 + 0.3 > 94)",
      short(TP_NEAR, ask_triggers=True).simulate()[0].exit_reason == "end_of_data")
LONG_SAME = [(100, 101, 99, 100), (100, 100.5, 96.9, 100), (100, 101, 99, 100)]
lng = Backtester(candles(LONG_SAME), Scripted(sig, 3.0, 6.0), execution=ExecutionModel(0.3, 0.0), ask_triggers=True)
check("long exits are unchanged by ask triggers", lng.simulate()[0].exit_reason == "stop_loss")

# ---- statistics ------------------------------------------------------------------
lo, hi = wilson_interval(21, 30)
check("Wilson interval, 70% of 30 trades", abs(lo - 0.521) < 0.01 and abs(hi - 0.834) < 0.01)
lo, hi = wilson_interval(210, 300)
check("Wilson interval narrows with 300 trades", abs(lo - 0.647) < 0.01 and abs(hi - 0.748) < 0.01)
check("Wilson interval with no trades", wilson_interval(0, 0) == (None, None))

# ---- the full report on a constructed run ----------------------------------------
# 6 blocks, each: [signal bar, ambiguous bar, spacer]. 1-minute data settles 3
# of them as target-first, and leaves 3 unresolved.
rows, days = [], []
for k in range(6):
    day = f"2025-02-{k + 3:02d}"
    block = [(100, 101, 99, 100), (100, 107, 96, 105), (105, 106, 104, 105)]
    start = pd.Timestamp(f"{day} 09:55", tz="UTC")
    times = pd.date_range(start, periods=3, freq="5min")
    days.append((day, k % 2 == 0))
    rows.append((times, block))
frame = pd.concat([pd.DataFrame({
    "timestamp": times, "open": [b[0] for b in block], "high": [b[1] for b in block],
    "low": [b[2] for b in block], "close": [b[3] for b in block]}) for times, block in rows],
    ignore_index=True)
for day, settled in days:
    if settled:   # target first
        write_minutes(day, [("10:00", 100, 100.5, 99.5, 100), ("10:01", 100, 107, 100, 106.5), ("10:02", 106.5, 106.6, 96, 97)])
    else:         # both inside one minute
        write_minutes(day, [("10:00", 100, 107, 96, 100)])
r = resolver()
bt = Backtester(frame, Scripted([1, 1, 0] * 6, 3.0, 6.0), resolver=r)
bt.simulate()
rep = accuracy_report(bt, r)
sc = rep["scenarios"]
check("6 ambiguous exits, 3 settled, 3 unresolved",
      rep["ambiguous_exits"] == 6 and rep["resolved_with_1m"] == 3 and rep["unresolved"] == 3)
check("central: only the settled ones win", sc["central"]["wins"] == 3)
check("pessimistic keeps the unresolved as losses", sc["pessimistic"]["wins"] == 3)
check("optimistic gives the unresolved to you", sc["optimistic"]["wins"] == 6)
check("engine win-rate range spans 50% to 100%", rep["win_rate_engine_range"] == [50.0, 100.0])
check("engine verdict is wide for a 50 point spread", rep["verdict"] == "wide")
check("and the sample is flagged separately", rep["sample"] == "too few trades")
check("band is wider than the statistical interval alone",
      rep["win_rate_band"][0] <= rep["win_rate_ci95"][0] and rep["win_rate_band"][1] >= rep["win_rate_ci95"][1])
check("central result is not changed by the report", len(bt.trades) == 6 and sum(t.pnl > 0 for t in bt.trades) == 3)
check("1-minute data usage is reported", rep["one_minute_data"]["days_loaded"] == 6)

# ---- no ambiguity at all -> tight ---------------------------------------------------
clean = [(100, 101, 99, 100), (100, 110, 99, 109), (109, 110, 108, 109)] * 40
bt = Backtester(candles(clean), Scripted([1, 1, 0] * 40, 3.0, 6.0))
bt.simulate()
rep = accuracy_report(bt)
check("no ambiguity: tight verdict, identical scenarios",
      rep["verdict"] == "tight" and rep["sample"] == "small sample" and rep["ambiguous_exits"] == 0
      and rep["win_rate_engine_range"][0] == rep["win_rate_engine_range"][1])


# ======================================================================
# Ground truth at scale: engine + 1-minute drill-down vs an independent
# minute-by-minute simulator, on a path built from the 1-minute data.
# ======================================================================
import numpy as np

rng = np.random.default_rng(5)
minutes = pd.date_range("2025-03-03 00:00", periods=8 * 1440, freq="1min", tz="UTC")
step = rng.normal(0, 0.9, len(minutes))
close_1m = 2000 + step.cumsum()
open_1m = np.concatenate([[close_1m[0]], close_1m[:-1]])
high_1m = np.maximum(open_1m, close_1m) + rng.uniform(0, 0.15, len(minutes))
low_1m = np.minimum(open_1m, close_1m) - rng.uniform(0, 0.15, len(minutes))
one = pd.DataFrame({"t": minutes, "o": open_1m, "h": high_1m, "l": low_1m, "c": close_1m})

for day, chunk in one.groupby(one.t.dt.date.astype(str)):
    write_minutes(day, [(f"{r.t:%H:%M}", r.o, r.h, r.l, r.c) for r in chunk.itertuples()])

grp = np.arange(len(one)) // 5
five = pd.DataFrame({
    "timestamp": one.t.iloc[::5].to_numpy(),
    "open": one.o.groupby(grp).first().to_numpy(), "high": one.h.groupby(grp).max().to_numpy(),
    "low": one.l.groupby(grp).min().to_numpy(), "close": one.c.groupby(grp).last().to_numpy(),
})
five["timestamp"] = pd.to_datetime(five["timestamp"], utc=True)

STOP, TARGET = 0.8, 1.1
entries = np.where(rng.random(len(five)) < 0.2, rng.choice([-1, 1], len(five)), 0)


class Events:
    def generate_signals(self, data):
        out = pd.DataFrame(index=data.index)
        out["entry"] = entries
        out["stop_distance"] = STOP
        out["target_distance"] = TARGET
        return out


def reference(five, one):
    """Independent minute-by-minute simulator (same conventions)."""
    trades, pos, pending = [], None, None
    for j, row in enumerate(five.itertuples()):
        if pending is not None and pos is None:
            side = pending
            o = row.open
            stop = o - STOP if side == 1 else o + STOP
            take = o + TARGET if side == 1 else o - TARGET
            pos = (side, stop, take, j)
        pending = None
        if pos is not None:
            side, stop, take, born = pos
            window = one[(one.t >= row.timestamp) & (one.t < row.timestamp + pd.Timedelta(minutes=5))]
            for m in window.itertuples():
                if side == 1:
                    sh, th = m.l <= stop, m.h >= take
                else:
                    sh, th = m.h >= stop, m.l <= take
                if sh or th:
                    # same-minute tie -> stop (matches the engine's fallback)
                    # a candle opening beyond a level fills there (5-min open rule)
                    reason = "stop_loss" if sh else "take_profit"
                    trades.append((born, j, reason))
                    pos = None
                    break
        if j < len(five) - 1 and entries[j] != 0:
            pending = int(entries[j])
    return trades


engine_bt = Backtester(five, Events(), resolver=resolver(), ambiguity="stop_first")
engine = [(five.index[five.timestamp == t.entry_time][0], five.index[five.timestamp == t.exit_time][0], t.exit_reason)
          for t in engine_bt.simulate() if t.exit_reason != "end_of_data"]
truth = reference(five, one)

amb = sum(m["ambiguous"] for m in engine_bt.trade_meta)
res = sum(m["ambiguous"] and m["resolved"] for m in engine_bt.trade_meta)
check(f"scale test has many ambiguous exits ({amb}) and many settled by 1-minute data ({res})", amb >= 20 and res >= 10)
check(f"engine + drill-down matches the minute-by-minute simulator on all {len(truth)} trades", engine == truth)

no_drill = Backtester(five, Events(), ambiguity="stop_first")
blind = [(five.index[five.timestamp == t.entry_time][0], five.index[five.timestamp == t.exit_time][0], t.exit_reason)
         for t in no_drill.simulate() if t.exit_reason != "end_of_data"]
check("without drill-down the engine DOES disagree with the truth (test is discriminating)", blind != truth)

rep = accuracy_report(engine_bt, resolver())
check("report at scale: engine range brackets the central win rate",
      rep["win_rate_engine_range"][0] <= rep["scenarios"]["central"]["win_rate"] <= rep["win_rate_engine_range"][1])
check("report at scale: most ambiguity settled by 1-minute data", rep["resolved_with_1m"] >= 0.8 * rep["ambiguous_exits"])

print("\nAll accuracy tests passed.")
