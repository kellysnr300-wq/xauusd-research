"""Tests for the strategy-output normalizer and the flexible strategy loader.

Usage:
    python -m src.test_normalize
"""

import numpy as np
import pandas as pd

from src import registry
from src.backtester import Backtester
from src.normalize import StrategyOutputError, normalize_output


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def raises(fn, text=""):
    try:
        fn()
    except StrategyOutputError as error:
        return text.lower() in str(error).lower()
    return False


def make_data(rows):
    times = pd.date_range("2025-01-06", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": times,
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
    })


flat = [(100, 101, 99, 100)] * 6
data = make_data(flat)
N = len(data)

# ---- input shapes --------------------------------------------------------
r = normalize_output(pd.Series([0, 1, 1, 0, -1, 0]), data)
check("Series -> signal", r.mode == "state" and list(r.frame["signal"]) == [0, 1, 1, 0, -1, 0])
check("list input", list(normalize_output([1, 0, 0, 0, 0, 0], data).frame["signal"])[0] == 1)
check("numpy input", normalize_output(np.array([0, 0, -1, 0, 0, 0]), data).frame["signal"].iloc[2] == -1)
check("bool input", list(normalize_output(pd.Series([True, False] * 3), data).frame["signal"]) == [1, 0] * 3)
r = normalize_output(pd.Series(["long", "short", "flat", None, "BUY", "sell"]), data)
check("word signals", list(r.frame["signal"]) == [1, -1, 0, 0, 1, -1])
r = normalize_output(pd.Series([0.7, -2.5, 0, np.nan, 1, 0]), data)
check("floats -> sign, with warning", list(r.frame["signal"]) == [1, -1, 0, 0, 1, 0] and len(r.warnings) == 1)
r = normalize_output(pd.DataFrame({"Position": [0, 1, 0, 0, 0, 0]}), data)
check("alias + case-insensitive columns", "signal" in r.frame.columns)
r = normalize_output(pd.DataFrame({"signal": [1] * N, "debug": range(N)}), data)
check("unknown columns ignored with note", any("debug" in w for w in r.warnings))
shifted = pd.DataFrame({"signal": [1] * N}, index=range(10, 10 + N))
check("misaligned index re-aligned", any("index" in w.lower() for w in normalize_output(shifted, data).warnings))

# ---- errors --------------------------------------------------------------
check("wrong length", raises(lambda: normalize_output([1, 0], data), "2 rows for 6"))
check("wrong type", raises(lambda: normalize_output({"a": 1}, data), "DataFrame"))
check("None output", raises(lambda: normalize_output(None, data), "NoneType"))
check("no direction column", raises(lambda: normalize_output(pd.DataFrame({"x": [1] * N}), data), "found columns"))
check("both signal and entry", raises(lambda: normalize_output(pd.DataFrame({"signal": [0] * N, "entry": [0] * N}), data), "not both"))
check("bad words", raises(lambda: normalize_output(pd.Series(["maybe"] * N), data), "unrecognised"))
check("infinite values", raises(lambda: normalize_output(pd.Series([np.inf] + [0] * 5), data), "infinite"))
check("negative stop distance", raises(lambda: normalize_output(pd.DataFrame({"signal": [1] * N, "stop_distance": [-1.0] * N}), data), "positive"))
ev = lambda **cols: normalize_output(pd.DataFrame({"entry": [1, 0, 0, 0, 0, 0], **cols}), data)
check("entry without stop", raises(lambda: ev(target_r=[3.0] * N), "no stop"))
check("entry without target", raises(lambda: ev(stop_price=[98.0] * N), "no target"))

# ---- distance shorthands -------------------------------------------------
r = normalize_output(pd.DataFrame({"signal": [1] * N, "stop_pct": [0.5] * N, "target_pct": [1.0] * N}), data)
check("pct -> distance", np.allclose(r.frame["stop_distance"], 0.5) and np.allclose(r.frame["target_distance"], 1.0))

atr_rows = [(100, 102, 98, 100)] * 30
atr_data = make_data(atr_rows)
r = normalize_output(pd.DataFrame({"signal": [1] * 30, "stop_atr": [1.5] * 30}), atr_data)
check("ATR -> distance (TR=4 -> 6)", np.isclose(r.frame["stop_distance"].iloc[-1], 6.0))
check("ATR warm-up signals ignored", list(r.frame["signal"].iloc[:13]) == [0] * 13 and r.frame["signal"].iloc[13] == 1 and any("warm-up" in w for w in r.warnings))
r = ev(stop_distance=[2.0] * N, rr=[3.0] * N)
check("rr alias -> target_r, event mode", r.mode == "event" and "target_r" in r.frame.columns)

# ---- engine integration: every exit style resolves exactly -------------------
class Fixed:
    def __init__(self, out): self.out = out
    def generate_signals(self, d): return self.out(d)

def trade(rows, out):
    d = make_data(rows)
    bt = Backtester(d, Fixed(lambda _d: out))
    return bt, bt.simulate()

up = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 108, 99, 107), (107, 108, 106, 107)]
bt, t = trade(up, pd.DataFrame({"entry": [1, 0, 0, 0], "stop_distance": [2.0] * 4, "target_r": [3.0] * 4}))
check("event: stop_distance + target_r exact 3R", t[0].exit_reason == "take_profit" and abs(t[0].pnl - 6.0) < 1e-9)
bt, t = trade(up, pd.DataFrame({"entry": [1, 0, 0, 0], "stop_price": [98.0] * 4, "target_price": [105.0] * 4}))
check("event: absolute stop and target prices", abs(t[0].pnl - 5.0) < 1e-9)
bt, t = trade(up, pd.DataFrame({"entry": [1, 0, 0, 0], "stop_pct": [2.0] * 4, "target_pct": [5.0] * 4}))
check("event: percent stop/target", abs(t[0].pnl - 5.0) < 1e-9)
bt, t = trade(up, pd.DataFrame({"signal": [1, 1, 1, 1], "stop_price": [98.0] * 4, "target_distance": [4.0] * 4}))
check("state mode: stop_price + target_distance", t[0].exit_reason == "take_profit" and abs(t[0].pnl - 4.0) < 1e-9)
bt, t = trade(up, pd.DataFrame({"signal": [1, 1, 1, 1], "stop_price": [200.0] * 4}))
check("wrong-side stop skipped and counted", len(t) == 0 and bt.skipped_entries >= 1)
bt, t = trade(up, [1, 1, 0, 0])
check("plain list works end to end", len(t) == 1)

# ---- flexible loader: function-style strategies ------------------------------
import tempfile
from pathlib import Path
registry.STRATEGY_DIR = Path(tempfile.mkdtemp())
code_fn = '''import pandas as pd

def generate_signals(data, length=3):
    average = data["close"].rolling(length).mean()
    return (data["close"] > average).astype(int)
'''
registry.save_strategy("Fn Strategy", "Other", "", {"length": 3}, code_fn)
strat = registry.build_strategy("fn_strategy", {"length": 3})
check("function strategy loads", hasattr(strat, "generate_signals"))
d = make_data([(100 + i, 101 + i, 99 + i, 100 + i) for i in range(20)])
bt = Backtester(d, strat)
check("function strategy runs in engine", len(bt.simulate()) >= 1)

code_helpers = '''def _helper(x):
    return x

def strategy(data):
    return [1] * len(data)

def other(x):
    return x
'''
registry.save_strategy("Fn Named", "Other", "", {}, code_helpers)
check("named function wins over other functions", registry.build_strategy("fn_named", {}).generate_signals(d) == [1] * len(d))

code_two = '''def a(data):
    return [0] * len(data)

def b(data):
    return [0] * len(data)
'''
registry.save_strategy("Fn Ambiguous", "Other", "", {}, code_two)
try:
    registry.build_strategy("fn_ambiguous", {})
    ok = False
except ValueError as error:
    ok = "No strategy found" in str(error)
check("ambiguous functions give a clear error", ok)

print("\nAll normalizer tests passed.")
