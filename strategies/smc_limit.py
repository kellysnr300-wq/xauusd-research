import numpy as np
import pandas as pd

from src.smc import (
    KILLZONES,
    align_htf,
    daily_levels,
    fvg,
    in_sessions,
    order_blocks,
    session_levels,
    structure,
    swings,
    sweeps,
)
from src.strategies import Strategy
from src.timeframes import TIMEFRAMES, resample_ohlc


class SmcLimit(Strategy):
    """SMC with LIMIT entries resting at the zone (not at the next open).

    While an unused order block (or fair value gap) is alive below (above)
    price, a limit order is re-posted every candle at the zone edge, for the
    next candle only, and only when that next candle is inside a killzone and
    the higher-timeframe trend agrees. The engine fills it the moment price
    trades to the level, at that price. Stop beyond the zone, target
    target_r x risk (exact, because the entry price is known in advance).

    entry_at: 0 = zone edge nearest price, 0.5 = middle, 1 = far edge.
    source:   "ob" (order blocks) or "fvg" (fair value gaps).
    """

    def __init__(
        self,
        source="ob",
        swing=2,
        kinds="both",
        lookback=5,
        zone="range",
        max_age=200,
        min_gap=0.5,
        entry_at=0.0,
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
        self.source = str(source).lower()
        self.swing = int(swing)
        self.kinds = str(kinds)
        self.lookback = int(lookback)
        self.zone = str(zone)
        self.max_age = int(max_age)
        self.min_gap = float(min_gap)
        self.entry_at = float(entry_at)
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

        if self.source not in ("ob", "fvg"):
            raise ValueError("source must be 'ob' or 'fvg'")
        if not 0.0 <= self.entry_at <= 1.0:
            raise ValueError("entry_at must be between 0 and 1")
        if self.htf != "none" and self.htf not in TIMEFRAMES:
            raise ValueError(f"htf must be 'none' or one of {list(TIMEFRAMES)}")
        for name in self.killzones:
            if name not in KILLZONES:
                raise ValueError(f"Unknown killzone '{name}'. Use {list(KILLZONES)}")
        for name in self.sweep_levels:
            if name not in (*KILLZONES, "pd", "pw", "swing"):
                raise ValueError(
                    f"Unknown sweep level '{name}'. Use asia, london, ny_am, "
                    "ny_pm, pd, pw or swing")

    @staticmethod
    def _names(text):
        text = str(text or "")
        if text.strip().lower() in ("", "none"):
            return []
        return [p.strip().lower() for p in text.split(",") if p.strip()]

    # ------------------------------------------------------------------

    @staticmethod
    def _step(data):
        stamps = pd.DatetimeIndex(data["timestamp"])[:200]
        steps = stamps[1:] - stamps[:-1]
        steps = steps[steps > pd.Timedelta(0)]
        return steps.min() if len(steps) else pd.Timedelta(minutes=5)

    def _htf_bias(self, data, step):
        if self.htf == "none":
            return None, "off"
        if step / pd.Timedelta(minutes=1) >= TIMEFRAMES[self.htf]["minutes"]:
            return None, "off, data already that slow"
        higher = resample_ohlc(data, self.htf)
        higher["trend"] = structure(higher, n=self.htf_swing)["trend"]
        return align_htf(data, higher, self.htf, ["trend"])["htf_trend"].to_numpy(), "on"

    def _recent_sweeps(self, data):
        count = len(data)
        low_hits = np.zeros(count, dtype=bool)
        high_hits = np.zeros(count, dtype=bool)
        for name in self.sweep_levels:
            if name in KILLZONES:
                lv = session_levels(data, name)
                found = sweeps(data, lv[f"prev_{name}_high"], lv[f"prev_{name}_low"])
            elif name in ("pd", "pw"):
                lv = daily_levels(data)
                hi, lo = ("pdh", "pdl") if name == "pd" else ("pwh", "pwl")
                found = sweeps(data, lv[hi], lv[lo])
            else:
                sw = swings(data, self.swing)
                found = sweeps(data, sw["last_swing_high"], sw["last_swing_low"])
            low_hits |= found["sweep_low"].to_numpy()
            high_hits |= found["sweep_high"].to_numpy()

        def recent(flags):
            return (pd.Series(flags.astype(float))
                    .rolling(self.sweep_window, min_periods=1).max()
                    .to_numpy().astype(bool))
        return recent(low_hits), recent(high_hits)

    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        count = len(data)
        close = data["close"].to_numpy(dtype=float)
        step = self._step(data)

        if self.source == "ob":
            zones = order_blocks(
                data, n=self.swing, lookback=self.lookback, zone=self.zone,
                kinds=self.kinds, max_age=self.max_age)
            p = "ob"
        else:
            zones = fvg(data, min_gap=self.min_gap, max_age=self.max_age)
            p = "fvg"

        b_top = zones[f"{p}_bull_top"].to_numpy()
        b_bot = zones[f"{p}_bull_bottom"].to_numpy()
        s_top = zones[f"{p}_bear_top"].to_numpy()
        s_bot = zones[f"{p}_bear_bottom"].to_numpy()

        long_ok = ~np.isnan(b_top) & self.allow_long
        short_ok = ~np.isnan(s_top) & self.allow_short
        funnel = {"1 candles with a live zone": int((long_ok | short_ok).sum())}

        # The order will be live during the NEXT candle: test that candle's time.
        if self.killzones:
            ahead = data.copy()
            ahead["timestamp"] = pd.DatetimeIndex(data["timestamp"]) + step
            zone_flags = in_sessions(ahead)
            active = np.zeros(count, dtype=bool)
            for name in self.killzones:
                active |= zone_flags[name].to_numpy()
            long_ok, short_ok = long_ok & active, short_ok & active
            label = f"2 next candle in killzone ({','.join(self.killzones)})"
        else:
            label = "2 killzone (off)"
        funnel[label] = int((long_ok | short_ok).sum())

        bias, state = self._htf_bias(data, step)
        if bias is not None:
            long_ok, short_ok = long_ok & (bias == 1), short_ok & (bias == -1)
        label = (f"3 with {self.htf} trend (on)" if state == "on"
                 else f"3 higher-timeframe trend ({state})")
        funnel[label] = int((long_ok | short_ok).sum())

        if self.require_sweep:
            recent_low, recent_high = self._recent_sweeps(data)
            long_ok, short_ok = long_ok & recent_low, short_ok & recent_high
            label = "4 after a liquidity sweep"
        else:
            label = "4 liquidity sweep (off)"
        funnel[label] = int((long_ok | short_ok).sum())

        price_long = b_top - self.entry_at * (b_top - b_bot)
        price_short = s_bot + self.entry_at * (s_top - s_bot)
        stop_long = b_bot - self.buffer
        stop_short = s_top + self.buffer
        risk_long = price_long - stop_long
        risk_short = stop_short - price_short

        long_ok = long_ok & (price_long < close) & (risk_long >= self.min_risk) & (risk_long <= self.max_risk)
        short_ok = short_ok & (price_short > close) & (risk_short >= self.min_risk) & (risk_short <= self.max_risk)
        both = long_ok & short_ok
        long_ok, short_ok = long_ok & ~both, short_ok & ~both
        funnel["5 valid price and risk = orders posted"] = int((long_ok | short_ok).sum())
        self.diagnostics = funnel

        entry = np.where(long_ok, 1, np.where(short_ok, -1, 0))
        price = np.where(long_ok, price_long, np.where(short_ok, price_short, np.nan))
        stop = np.where(long_ok, stop_long, np.where(short_ok, stop_short, np.nan))

        return pd.DataFrame({
            "entry": entry,
            "entry_price": price,
            "stop_price": stop,
            "target_r": np.where(entry != 0, self.target_r, np.nan),
            "valid_for": np.where(entry != 0, 1.0, np.nan),
        }, index=data.index)
