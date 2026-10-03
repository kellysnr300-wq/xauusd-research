import pandas as pd

from src.strategies import Strategy


class MovingAverageCross(Strategy):
    """Baseline: long when fast SMA > slow SMA, short when below.

    Uses only closes up to and including the current candle, so the
    signal is known at that candle's close (the engine then trades at
    the next candle's open). Flat until the slow average has enough data.
    """

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast must be smaller than slow")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]

        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()

        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals.loc[fast_ma > slow_ma, "signal"] = 1
        signals.loc[fast_ma < slow_ma, "signal"] = -1

        return signals
