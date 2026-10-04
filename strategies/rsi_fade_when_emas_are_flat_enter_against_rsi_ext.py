import pandas as pd
import numpy as np

from src.strategies import Strategy


class EmaRsiPullback5m(Strategy):
    """
    5m mean reversion: fade RSI extremes in a flat market.

    Trend strategies on this data printed PF \~0.60-0.65.
    This does the opposite:
      - Only trade when |EMA20 - EMA50| / ATR is small (chop)
      - Long when RSI crosses up from below rsi_long (default 25)
      - Short when RSI crosses down from above rsi_short (default 75)
      - Tight target (1.0 ATR), stop 1.2 ATR  (need WR > \~55%)
      - Short max hold; cooldown after exit
    """

    def __init__(
        self,
        length=20,
        ema_fast=None,
        ema_slow=50,
        rsi_len=14,
        rsi_long=25.0,
        rsi_short=75.0,
        atr_len=14,
        flat_atr_max=1.5,
        stop_atr=1.2,
        target_atr=1.0,
        cooldown_bars=5,
        max_hold_bars=12,
        allow_short=True,
        **kwargs,
    ):
        self.ema_fast = length if ema_fast is None else ema_fast
        self.ema_slow = ema_slow
        self.rsi_len = rsi_len
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.atr_len = atr_len
        self.flat_atr_max = flat_atr_max
        self.stop_atr = stop_atr
        self.target_atr = target_atr
        self.cooldown_bars = cooldown_bars
        self.max_hold_bars = max_hold_bars
        self.allow_short = allow_short

    def _ema(self, s, n):
        return s.ewm(span=n, adjust=False).mean()

    def _rsi(self, close, n):
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        avg_up = up.ewm(alpha=1 / n, adjust=False).mean()
        avg_down = down.ewm(alpha=1 / n, adjust=False).mean()
        rs = avg_up / avg_down.replace(0.0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    def _atr(self, data, n):
        prev_close = data["close"].shift(1)
        tr1 = data["high"] - data["low"]
        tr2 = (data["high"] - prev_close).abs()
        tr3 = (data["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / n, adjust=False).mean()

    def generate_signals(self, data):
        close = data["close"]
        n = len(data)

        ema_fast = self._ema(close, self.ema_fast)
        ema_slow = self._ema(close, self.ema_slow)
        rsi = self._rsi(close, self.rsi_len)
        atr = self._atr(data, self.atr_len)

        # Flat market: EMAs are close relative to volatility
        ema_gap = (ema_fast - ema_slow).abs()
        is_flat = ema_gap <= (self.flat_atr_max * atr)

        rsi_prev = rsi.shift(1)
        rsi_cross_up = (rsi_prev <= self.rsi_long) & (rsi > self.rsi_long)
        rsi_cross_dn = (rsi_prev >= self.rsi_short) & (rsi < self.rsi_short)

        long_trig = is_flat & rsi_cross_up
        short_trig = is_flat & rsi_cross_dn
        if not self.allow_short:
            short_trig = pd.Series(False, index=data.index)

        signal = np.zeros(n, dtype=int)
        pos = 0
        held = 0
        cooldown = 0

        long_a = long_trig.fillna(False).to_numpy()
        short_a = short_trig.fillna(False).to_numpy()
        flat_a = is_flat.fillna(False).to_numpy()

        for i in range(n):
            if cooldown > 0:
                cooldown -= 1

            if pos != 0:
                held += 1
                # Exit if market starts trending or time is up
                left_chop = not flat_a[i]
                timed_out = held >= self.max_hold_bars
                if left_chop or timed_out:
                    pos = 0
                    held = 0
                    cooldown = self.cooldown_bars
                    signal[i] = 0
                else:
                    signal[i] = pos
                continue

            if cooldown > 0:
                continue

            if long_a[i]:
                pos = 1
                held = 1
                signal[i] = 1
            elif short_a[i]:
                pos = -1
                held = 1
                signal[i] = -1

        out = pd.DataFrame(index=data.index)
        out["signal"] = signal
        out["stop_atr"] = self.stop_atr
        out["target_atr"] = self.target_atr
        out["target_r"] = self.target_atr / self.stop_atr
        return out