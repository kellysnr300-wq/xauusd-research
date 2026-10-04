import numpy as np
import pandas as pd

from src.smc import (
    KILLZONES,
    align_htf,
    daily_levels,
    in_sessions,
    order_blocks,
    session_levels,
    structure,
    swings,
    sweeps,
)
from src.strategies import Strategy
from src.timeframes import TIMEFRAMES, resample_ohlc


class SmcV1(Strategy):
    """SMC v1: retest of the order block that broke structure.

    Pipeline (each filter can be switched off; `self.diagnostics` counts
    how many setups survive each step, shown in the dashboard):

    1. A BOS / CHoCH creates an order block (see src/smc.py).
    2. Price retests the block      -> raw setup.
    3. Retest happens in a killzone (killzones="" turns this off).
    4. Higher-timeframe trend agrees (htf="none" turns this off).
    5. A liquidity sweep happened recently (require_sweep, off by default).
    6. The stop was not already hit on the retest candle and the risk is
       within [min_risk, max_risk].

    Entry at the next open, stop beyond the block, target = target_r x risk.
    """

    def __init__(
        self,
        swing=2,
        kinds="both",
        lookback=5,
        zone="range",
        max_age=200,
        htf="1h",
        htf_swing=1,
        killzones="london,ny_am",
        require_sweep=False,
        sweep_levels="asia,pd,swing",
        sweep_window=24,
        buffer=0.3,
        target_r=3.0,
        min_risk=1.0,
        max_risk=25.0,
        allow_long=True,
        allow_short=True,
    ):
        self.swing = int(swing)
        self.kinds = str(kinds)
        self.lookback = int(lookback)
        self.zone = str(zone)
        self.max_age = int(max_age)
        self.htf = str(htf).lower()
        self.htf_swing = int(htf_swing)
        self.killzones = self._names(killzones)
        self.require_sweep = bool(require_sweep)
        self.sweep_levels = self._names(sweep_levels)
        self.sweep_window = max(int(sweep_window), 1)
        self.buffer = float(buffer)
        self.target_r = float(target_r)
        self.min_risk = float(min_risk)
        self.max_risk = float(max_risk)
        self.allow_long = bool(allow_long)
        self.allow_short = bool(allow_short)
        self.diagnostics = {}

        if self.htf != "none" and self.htf not in TIMEFRAMES:
            raise ValueError(
                f"htf must be 'none' or one of {list(TIMEFRAMES)}")
        for name in self.killzones:
            if name not in KILLZONES:
                raise ValueError(
                    f"Unknown killzone '{name}'. Use {list(KILLZONES)}")
        for name in self.sweep_levels:
            if name not in (*KILLZONES, "pd", "pw", "swing"):
                raise ValueError(
                    f"Unknown sweep level '{name}'. Use asia, london, "
                    "ny_am, ny_pm, pd, pw or swing")

    @staticmethod
    def _names(text):
        text = str(text or "")
        if text.strip().lower() in ("", "none"):
            return []
        return [part.strip().lower() for part in text.split(",") if part.strip()]

    # ------------------------------------------------------------------

    def _htf_bias(self, data):
        """HTF trend per candle, or (None, reason) if not applicable."""
        if self.htf == "none":
            return None, "off"

        stamps = pd.DatetimeIndex(data["timestamp"])[:200]
        steps = stamps[1:] - stamps[:-1]
        steps = steps[steps > pd.Timedelta(0)]
        base_minutes = steps.min() / pd.Timedelta(minutes=1) if len(steps) else None

        if base_minutes is None or base_minutes >= TIMEFRAMES[self.htf]["minutes"]:
            return None, "off, data already that slow"

        higher = resample_ohlc(data, self.htf)
        higher["trend"] = structure(higher, n=self.htf_swing)["trend"]
        aligned = align_htf(data, higher, self.htf, ["trend"])
        return aligned["htf_trend"].to_numpy(), "on"

    def _recent_sweeps(self, data):
        count = len(data)
        low_hits = np.zeros(count, dtype=bool)
        high_hits = np.zeros(count, dtype=bool)

        for name in self.sweep_levels:
            if name in KILLZONES:
                levels = session_levels(data, name)
                found = sweeps(data, levels[f"prev_{name}_high"],
                               levels[f"prev_{name}_low"])
            elif name in ("pd", "pw"):
                levels = daily_levels(data)
                high_name, low_name = ("pdh", "pdl") if name == "pd" else ("pwh", "pwl")
                found = sweeps(data, levels[high_name], levels[low_name])
            else:
                structure_swings = swings(data, self.swing)
                found = sweeps(data, structure_swings["last_swing_high"],
                               structure_swings["last_swing_low"])
            low_hits |= found["sweep_low"].to_numpy()
            high_hits |= found["sweep_high"].to_numpy()

        def recent(flags):
            rolled = pd.Series(flags.astype(float)).rolling(
                self.sweep_window, min_periods=1).max()
            return rolled.to_numpy().astype(bool)

        return recent(low_hits), recent(high_hits)

    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        blocks = order_blocks(
            data, n=self.swing, lookback=self.lookback, zone=self.zone,
            kinds=self.kinds, max_age=self.max_age,
        )

        close = data["close"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)

        long_hit = blocks["ob_bull_retest"].to_numpy() & self.allow_long
        short_hit = blocks["ob_bear_retest"].to_numpy() & self.allow_short
        stop_long = blocks["ob_bull_retest_bottom"].to_numpy() - self.buffer
        stop_short = blocks["ob_bear_retest_top"].to_numpy() + self.buffer

        funnel = {"1 order block retests": int(long_hit.sum() + short_hit.sum())}

        # 3. killzone
        if self.killzones:
            zones = in_sessions(data)
            active = np.zeros(len(data), dtype=bool)
            for name in self.killzones:
                active |= zones[name].to_numpy()
            long_hit, short_hit = long_hit & active, short_hit & active
            label = f"2 inside killzone ({','.join(self.killzones)})"
        else:
            label = "2 killzone (off)"
        funnel[label] = int(long_hit.sum() + short_hit.sum())

        # 4. higher-timeframe bias
        bias, state = self._htf_bias(data)
        if bias is not None:
            long_hit, short_hit = long_hit & (bias == 1), short_hit & (bias == -1)
        label = (f"3 with {self.htf} trend (on)" if state == "on"
                 else f"3 higher-timeframe trend ({state})")
        funnel[label] = int(long_hit.sum() + short_hit.sum())

        # 5. liquidity sweep
        if self.require_sweep:
            recent_low, recent_high = self._recent_sweeps(data)
            long_hit, short_hit = long_hit & recent_low, short_hit & recent_high
            label = "4 after a liquidity sweep"
        else:
            label = "4 liquidity sweep (off)"
        funnel[label] = int(long_hit.sum() + short_hit.sum())

        # 6. stop sanity and risk limits
        risk_long = close - stop_long
        risk_short = stop_short - close
        long_ok = (
            long_hit & (low > stop_long)
            & (risk_long >= self.min_risk) & (risk_long <= self.max_risk)
        )
        short_ok = (
            short_hit & (high < stop_short)
            & (risk_short >= self.min_risk) & (risk_short <= self.max_risk)
        )
        both = long_ok & short_ok
        long_ok, short_ok = long_ok & ~both, short_ok & ~both
        funnel["5 valid stop and risk = entries"] = int(
            long_ok.sum() + short_ok.sum())

        self.diagnostics = funnel

        entry = np.where(long_ok, 1, np.where(short_ok, -1, 0))
        stop = np.where(long_ok, stop_long, np.where(short_ok, stop_short, np.nan))

        return pd.DataFrame({
            "entry": entry,
            "stop_price": stop,
            "target_r": np.where(entry != 0, self.target_r, np.nan),
        }, index=data.index)
