import pandas as pd
import numpy as np

from src.strategies import Strategy


class EmaRsiPullback5m(Strategy):
    """
    5m compression -> momentum breakout.

    Replaces the failed RSI-pullback idea (PF \~0.65, WR \~24%).

    Entry long:
      - EMA20 > EMA50 (trend)
      - close > EMA20
      - current bar range > 1.2 * ATR  (expansion)
      - close in top 30% of the bar   (bullish close)
      - prior 3 bars each had range < ATR  (compression)
    Short is the mirror.

    Risk: stop 1.0 ATR, target 2.0 ATR. Cooldown + max hold.
    """

    def __init__(
        self,
        length=20,
        ema_fast=None,
        ema_slow=50,
        atr_len=14,
        expand_mult=1.2,
        compress_bars=3,
        close_loc=0.30,
        stop_atr=1.0,
        target_atr=2.0,
        cooldown_bars=4,
        max_hold_bars=24,
        allow_short=True,
        **kwargs,
    ):
        self.ema_fast = length if ema_fast is None else ema_fast
        self.ema_slow = ema_slow
        self.atr_len = atr_len
        self.expand_mult = expand_mult
        self.compress_bars = compress_bars
        self.close_loc = close_loc
        self.stop_atr = stop_atr
        self.target_atr = target_atr
        self.cooldown_bars = cooldown_bars
        self.max_hold_bars = max_hold_bars
        self.allow_short = allow_short

    def _ema(self, s, n):
        return s.ewm(span=n, adjust=False).mean()

    def _atr(self, data, n):
        prev_close = data["close"].shift(1)
        tr1 = data["high"] - data["low"]
        tr2 = (data["high"] - prev_close).abs()
        tr3 = (data["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / n, adjust=False).mean()

    def generate_signals(self, data):
        close = data["close"]
        high = data["high"]
        low = data["low"]
        n = len(data)

        ema_fast = self._ema(close, self.ema_fast)
        ema_slow = self._ema(close, self.ema_slow)
        atr = self._atr(data, self.atr_len)
        bar_range = high - low

        # Where the close sits inside the bar (0 = low, 1 = high)
        loc = (close - low) / bar_range.replace(0, np.nan)

        uptrend = (ema_fast > ema_slow) & (close > ema_fast)
        downtrend = (ema_fast < ema_slow) & (close < ema_fast)

        expansion = bar_range > (self.expand_mult * atr)

        # Prior bars were quiet (compression)
        quiet = bar_range < atr
        compression = quiet.shift(1)
        for k in range(2, self.compress_bars + 1):
            compression = compression & quiet.shift(k)

        long_trig = (
            uptrend
            & expansion
            & (loc >= (1.0 - self.close_loc))
            & compression
        )
        short_trig = (
            downtrend
            & expansion
            & (loc <= self.close_loc)
            & compression
        )
        if not self.allow_short:
            short_trig = pd.Series(False, index=data.index)

        signal = np.zeros(n, dtype=int)
        pos = 0
        held = 0
        cooldown = 0

        long_a = long_trig.fillna(False).to_numpy()
        short_a = short_trig.fillna(False).to_numpy()
        up_a = uptrend.fillna(False).to_numpy()
        dn_a = downtrend.fillna(False).to_numpy()

        for i in range(n):
            if cooldown > 0:
                cooldown -= 1

            if pos != 0:
                held += 1
                lost = (pos == 1 and not up_a[i]) or (pos == -1 and not dn_a[i])
                if lost or held >= self.max_hold_bars:
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