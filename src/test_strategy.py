import pandas as pd

from src.strategies import Strategy


class DummyStrategy(Strategy):
    """Simple strategy used only to test the backtesting engine."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0

        if len(signals) >= 2:
            signals.iloc[1, signals.columns.get_loc("signal")] = 1

        return signals
