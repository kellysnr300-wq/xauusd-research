import pandas as pd
import numpy as np

from src.strategies import Strategy


class EmaRsiPullback5m(Strategy):
    """
    5m EMA trend + RSI pullback (balanced).

    Prior v1: 17k trades, PF 0.63 (too loose).
    Prior v2: 0 trades (too tight: RSI 30/70 + slope + vol).
    This version: RSI 40/60, no slope filter, softer vol filter,
    cooldown + max hold to limit churn.
    """

    def __init__(
        self,
        length=20,
        ema_fast=None,
        ema_slow=50,
        rsi_len=14,
        rsi_long=40.0,
        rsi_short=60.0,
        atr_len=14,
        atr_median_len=50,
        spike_atr_mult=3.0,
        stop_atr=1.5,
        target_atr=2.5,
        cooldown_bars=3,
        max_hold_bars=36,
        allow_short=True,
        **kwargs,
    ):
        self.ema_fast = length if ema_fast is None else ema_fast
        self.ema_slow = ema_slow
        self.rsi_len = rsi_len
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.atr_len = atr_len
        self.atr_median_len = atr_median_len
        self.spike_atr_mult = spike_atr_mult
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
        high = data["high"]
        low = data["low"]
        n = len(data)

        ema_fast = self._ema(close, self.ema_fast)
        ema_slow = self._ema(close, self.ema_slow)
        rsi = self._rsi(close, self.rsi_len)
        atr = self._atr(data, self.atr_len)
        atr_med = atr.rolling(self.atr_median_len).median()
        bar_range = high - low

        uptrend = (ema_fast > ema_slow) & (close > ema_fast)
        downtrend = (ema_fast < ema_slow) & (close < ema_fast)

        rsi_prev = rsi.shift(1)
        rsi_cross_up = (rsi_prev <= self.rsi_long) & (rsi > self.rsi_long)
        rsi_cross_dn = (rsi_prev >= self.rsi_short) & (rsi < self.rsi_short)

        # Soft vol filter: skip only dead bars and extreme spikes
        enough_vol = atr >= (0.7 * atr_med)
        not_spike = bar_range <= (self.spike_atr_mult * atr)

        long_trig = uptrend & rsi_cross_up & enough_vol & not_spike
        short_trig = downtrend & rsi_cross_dn & enough_vol & not_spike
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
                lost_trend = (pos == 1 and not up_a[i]) or (
                    pos == -1 and not dn_a[i]
                )
                timed_out = held >= self.max_hold_bars
                if lost_trend or timed_out:
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