"""Tests for the look-ahead guard itself.

Usage:
    python -m src.test_lookahead
"""

import numpy as np
import pandas as pd

from src.lookahead import check_no_lookahead


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


rng = np.random.default_rng(4)
times = pd.date_range("2024-01-01 22:00", periods=9 * 7 * 288, freq="5min", tz="UTC")
wd, hr = times.weekday, times.hour
times = times[~(((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 22)) | (hr == 21))]
n = len(times)
close = 2000 + rng.normal(0, 0.7, n).cumsum()
o = np.concatenate([[close[0]], close[:-1]])
data = pd.DataFrame({"timestamp": times, "open": o,
                     "high": np.maximum(o, close) + 0.2, "low": np.minimum(o, close) - 0.2,
                     "close": close, "volume": 1})


def ahead(c, k):
    return np.concatenate([c[k:], np.full(k, np.nan)]) - c


class SparseEventCheater:
    """Acts rarely, peeking 3 candles ahead: the case a few broad cuts can miss."""
    def __init__(self, threshold=3.4):
        self.threshold = threshold

    def generate_signals(self, d):
        entry = np.where(ahead(d["close"].to_numpy(), 3) > self.threshold, 1, 0)
        return pd.DataFrame({"entry": entry, "stop_distance": 1.0, "target_r": 1.0}, index=d.index)


class SparseStateCheater:
    """A held position that flips rarely using a peek 4 candles ahead."""
    def generate_signals(self, d):
        future = ahead(d["close"].to_numpy(), 4)
        signal = pd.Series(np.where(future > 2.6, 1, np.where(future < -2.6, -1, np.nan))).ffill().fillna(0).astype(int)
        return pd.DataFrame({"signal": signal.to_numpy()}, index=d.index)


class GlobalStatCheater:
    """Uses the mean of the WHOLE sample."""
    def generate_signals(self, d):
        c = d["close"]
        return pd.DataFrame({"signal": (c > c.mean()).astype(int)}, index=d.index)


class HonestEvent:
    """Breakout of the previous 20-candle high: only the past."""
    def generate_signals(self, d):
        prior = d["high"].rolling(20).max().shift(1)
        entry = np.where(d["close"] > prior, 1, 0)
        return pd.DataFrame({"entry": entry, "stop_distance": 1.0, "target_r": 2.0}, index=d.index)


class HonestState:
    def generate_signals(self, d):
        c = d["close"]
        return pd.DataFrame({"signal": np.where(c > c.rolling(50).mean(), 1, -1)}, index=d.index)


entries = int((SparseEventCheater().generate_signals(data).entry != 0).sum())
print(f"      info: the sparse cheater acts on {entries} of {len(data)} candles ({entries / len(data):.2%})")
check("the sparse cheater really is sparse", 5 <= entries <= 0.02 * len(data))

old_style = check_no_lookahead(SparseEventCheater, data, targeted=0)
check("OLD behaviour (broad cuts only) misses the sparse cheater - the hole that was found", old_style == [])
new_style = check_no_lookahead(SparseEventCheater, data)
check("NEW behaviour catches it", len(new_style) == 1 and "LOOK-AHEAD" in new_style[0] and "decision at candle" in new_style[0])

state_old = check_no_lookahead(SparseStateCheater, data, targeted=0)
state_new = check_no_lookahead(SparseStateCheater, data)
check("a sparse state-mode peek is caught too", len(state_new) == 1 and "LOOK-AHEAD" in state_new[0])
print(f"      info: broad cuts alone on the state cheater: {'missed' if state_old == [] else 'caught'}")

check("whole-sample statistics are caught", "LOOK-AHEAD" in check_no_lookahead(GlobalStatCheater, data)[0])
check("an honest event strategy is NOT flagged", check_no_lookahead(HonestEvent, data) == [])
check("an honest state strategy is NOT flagged", check_no_lookahead(HonestState, data) == [])

for threshold in (2.8, 3.0, 3.4, 3.6):
    found = check_no_lookahead(lambda t=threshold: SparseEventCheater(t), data)
    check(f"sparse cheater caught at threshold {threshold}", len(found) == 1 and "LOOK-AHEAD" in found[0])

print("\nAll look-ahead guard tests passed.")
