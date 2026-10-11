"""Tests for the batch screening tool.

Usage:
    python -m src.test_screen
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import registry
from src.execution import ExecutionModel
from src.screen import leaderboard, screen
from src.timeframes import max_entry_gap_for


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


# A market with real structure: each return partly reverses the previous one.
rng = np.random.default_rng(8)
times = pd.date_range("2024-01-01 22:00", periods=9 * 7 * 288, freq="5min", tz="UTC")
wd, hr = times.weekday, times.hour
times = times[~(((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 22)) | (hr == 21))]
n = len(times)
noise = rng.normal(0, 0.7, n)
ret = np.zeros(n)
for i in range(1, n):
    ret[i] = -0.55 * ret[i - 1] + noise[i]
close = 2000 + ret.cumsum()
open_ = np.concatenate([[close[0]], close[:-1]])
data = pd.DataFrame({"timestamp": times, "open": open_,
                     "high": np.maximum(open_, close) + 0.1, "low": np.minimum(open_, close) - 0.1,
                     "close": close, "volume": 100})

registry.STRATEGY_DIR = Path(tempfile.mkdtemp())

HEADER = "import numpy as np\nimport pandas as pd\nfrom src.strategies import Strategy\n\n\n"
CODES = {
    "noskill": HEADER + '''class NoSkill(Strategy):
    """Random timing, but each row's randomness depends only on its position
    (a hash), never on how much data exists: a data-length-dependent random
    stream would itself be non-causal and rejected by the guard."""
    def generate_signals(self, d):
        i = np.arange(len(d))
        u1 = np.mod(np.sin(i * 12.9898 + 1.0) * 43758.5453, 1.0)
        u2 = np.mod(np.sin(i * 78.233 + 2.0) * 43758.5453, 1.0)
        entry = np.where(u1 < 0.03, np.where(u2 < 0.5, -1, 1), 0)
        return pd.DataFrame({"entry": entry, "stop_distance": 1.0, "target_r": 1.0}, index=d.index)
''',
    "skilled": HEADER + '''class Skilled(Strategy):
    """Fades the last candle's move: real edge in this market, no look-ahead."""
    def generate_signals(self, d):
        c = d["close"].to_numpy()
        last = np.concatenate([[0.0], c[1:] - c[:-1]])
        entry = np.where(last < -1.2, 1, np.where(last > 1.2, -1, 0))
        return pd.DataFrame({"entry": entry, "stop_distance": 1.0, "target_r": 1.0}, index=d.index)
''',
    "cheater": HEADER + '''class Cheater(Strategy):
    def generate_signals(self, d):
        c = d["close"].to_numpy()
        future = np.concatenate([c[3:], np.full(3, np.nan)]) - c
        entry = np.where(future > 1.0, 1, 0)
        return pd.DataFrame({"entry": entry, "stop_distance": 1.0, "target_r": 1.0}, index=d.index)
''',
    "broken": HEADER + '''class Broken(Strategy):
    def generate_signals(self, d):
        raise RuntimeError("boom")
''',
    "silent": HEADER + '''class Silent(Strategy):
    def generate_signals(self, d):
        return pd.DataFrame({"signal": np.zeros(len(d), dtype=int)}, index=d.index)
''',
    "garbage": HEADER + '''class Garbage(Strategy):
    def generate_signals(self, d):
        return {"a": 1}
''',
    "slow": HEADER + '''import time


class Slow(Strategy):
    def generate_signals(self, d):
        time.sleep(120)
        return pd.DataFrame({"signal": np.zeros(len(d), dtype=int)}, index=d.index)
''',
}
for sid, code in CODES.items():
    registry.save_strategy(sid, "Other", "", {}, code)

logs = []
rows = screen(data, list(CODES), runs=19, seed=2, execution=ExecutionModel(0.0, 0.0),
              max_gap=max_entry_gap_for("5m"), max_shift_days=10, timeout_s=10,
              min_trades=30, log=logs.append)
by = {r["id"]: r for r in rows}

check("every strategy gets a row, none stops the batch", len(rows) == len(CODES))
check("cheater is excluded by the look-ahead guard", by["cheater"]["status"] == "LOOK-AHEAD (excluded)")
check("crash is recorded, not fatal", by["broken"]["status"].startswith("error: RuntimeError"))
check("never-signalling strategy is reported", by["silent"]["status"] == "no trades")
check("unusable output is explained", by["garbage"]["status"].startswith("invalid output"))
check("a hanging strategy times out and the batch moves on",
      by["slow"]["status"].startswith("timeout") and by["slow"]["seconds"] < 25)
check("no-skill strategy is evaluated and NOT flagged",
      by["noskill"]["status"] == "evaluated" and by["noskill"]["p"] > 0.05)
print(f"      info: no-skill p = {by['noskill']['p']:.3f}, edge {by['noskill']['edge']:+.3f}/trade; "
      f"skilled p = {by['skilled']['p']:.3f}, edge {by['skilled']['edge']:+.3f}/trade")
check("a strategy with real, non-leaking skill is detected",
      by["skilled"]["status"] == "evaluated" and by["skilled"]["verdict"] == "BETTER than random timing")

text = leaderboard(rows)
check("leaderboard puts the best p first", text.index("skilled") < text.index("noskill"))
check("leaderboard lists what was not evaluated and why",
      "Not evaluated:" in text and "LOOK-AHEAD (excluded)" in text and "timeout" in text)
check("leaderboard names the strategy worth a longer check",
      "Worth a longer check" in text and "skilled" in text.split("Worth a longer check")[1])

relogs = []
again = screen(data, list(CODES), runs=19, seed=2, execution=ExecutionModel(0.0, 0.0),
               max_gap=max_entry_gap_for("5m"), max_shift_days=10, timeout_s=10,
               min_trades=30, done=[by["noskill"], by["skilled"]], log=relogs.append)
check("resume skips strategies that are already done",
      not any("noskill" in line or "skilled" in line for line in relogs) and len(again) == len(CODES))

print("\nAll screen tests passed.")
