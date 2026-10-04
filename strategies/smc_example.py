import numpy as np
import pandas as pd

from src.smc import fvg, in_sessions, structure
from src.strategies import Strategy


class SmcExample(Strategy):
    """Demo of the SMC blocks (NOT a tuned strategy).

    Trade WITH the trend set by the latest BOS/CHoCH, entering when price
    returns to an open fair value gap inside a killzone. Stop beyond the
    gap, target `target_r` times the risk.
    """

    def __init__(self, swing=2, min_gap=0.5, buffer=0.3, target_r=2.0,
                 killzones="london,ny_am", min_risk=1.0, max_risk=25.0):
        self.swing = int(swing)
        self.min_gap = float(min_gap)
        self.buffer = float(buffer)
        self.target_r = float(target_r)
        self.killzones = [z.strip() for z in str(killzones).split(",") if z.strip()]
        self.min_risk = float(min_risk)
        self.max_risk = float(max_risk)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        trend = structure(data, n=self.swing)["trend"].to_numpy()
        gaps = fvg(data, min_gap=self.min_gap)
        zones = in_sessions(data)

        active = np.zeros(len(data), dtype=bool)
        for name in self.killzones:
            active |= zones[name].to_numpy()

        close = data["close"].to_numpy(dtype=float)
        stop_long = gaps["fvg_bull_retest_bottom"].to_numpy() - self.buffer
        stop_short = gaps["fvg_bear_retest_top"].to_numpy() + self.buffer
        risk_long = close - stop_long
        risk_short = stop_short - close

        long_ok = (
            (trend == 1) & gaps["fvg_bull_retest"].to_numpy() & active
            & (risk_long >= self.min_risk) & (risk_long <= self.max_risk)
        )
        short_ok = (
            (trend == -1) & gaps["fvg_bear_retest"].to_numpy() & active
            & (risk_short >= self.min_risk) & (risk_short <= self.max_risk)
        )

        entry = np.where(long_ok, 1, np.where(short_ok, -1, 0))
        stop = np.where(long_ok, stop_long, np.where(short_ok, stop_short, np.nan))

        return pd.DataFrame({
            "entry": entry,
            "stop_price": stop,
            "target_r": np.where(entry != 0, self.target_r, np.nan),
        }, index=data.index)
