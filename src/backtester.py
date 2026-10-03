import pandas as pd

from src.execution import ExecutionModel
from src.portfolio import Portfolio
from src.strategies import Strategy
from src.trades import Trade


class Backtester:
    """Strategy-independent historical backtesting engine."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 10_000.0,
        execution: ExecutionModel | None = None,
    ):
        self.data = data.copy()
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)
        self.execution = execution or ExecutionModel()
        self.trades: list[Trade] = []

    def run(self) -> pd.DataFrame:
        """Run the strategy and return generated signals."""

        signals = self.strategy.generate_signals(self.data)

        if len(signals) != len(self.data):
            raise ValueError(
                "Strategy must return one signal row per candle."
            )

        return signals

    def record_trade(self, trade: Trade) -> None:
        """Record a completed trade and update portfolio."""

        self.trades.append(trade)
        self.portfolio.record_pnl(trade.pnl)
