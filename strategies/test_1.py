import pandas as pd

from src.strategies import Strategy


class MyStrategy(Strategy):
    """5m XAUUSD trend-pullback continuation strategy."""

    def __init__(
        self,
        length=20,
        slow_length=50,
        atr_length=14,
    ):
        self.length = length
        self.slow_length = slow_length
        self.atr_length = atr_length

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        # Fast EMA
        fast_ema = close.ewm(
            span=self.length,
            adjust=False
        ).mean()

        # Slow EMA
        slow_ema = close.ewm(
            span=self.slow_length,
            adjust=False
        ).mean()

        # True Range
        previous_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - previous_close).abs()
        tr3 = (low - previous_close).abs()

        true_range = pd.concat(
            [tr1, tr2, tr3],
            axis=1
        ).max(axis=1)

        # ATR
        atr = true_range.rolling(
            self.atr_length
        ).mean()

        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0

        # -------------------------
        # TREND
        # -------------------------

        bullish_trend = (
            (fast_ema > slow_ema) &
            (close > slow_ema)
        )

        bearish_trend = (
            (fast_ema < slow_ema) &
            (close < slow_ema)
        )

        # -------------------------
        # PULLBACK
        # -------------------------

        bullish_pullback = low <= fast_ema
        bearish_pullback = high >= fast_ema

        # -------------------------
        # MOMENTUM
        # -------------------------

        candle_range = high - low

        bullish_momentum = (
            (close > open_) &
            (close > fast_ema) &
            (candle_range >= 0.5 * atr)
        )

        bearish_momentum = (
            (close < open_) &
            (close < fast_ema) &
            (candle_range >= 0.5 * atr)
        )

        # -------------------------
        # ENTRY
        # -------------------------

        long_entry = (
            bullish_trend &
            bullish_pullback &
            bullish_momentum
        )

        short_entry = (
            bearish_trend &
            bearish_pullback &
            bearish_momentum
        )

        signals.loc[long_entry, "signal"] = 1
        signals.loc[short_entry, "signal"] = -1

        return signals