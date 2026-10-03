from dataclasses import dataclass


@dataclass
class Portfolio:
    """Tracks simulated account equity."""

    initial_capital: float
    realized_pnl: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0

    def __post_init__(self):
        self.peak_equity = self.initial_capital

    @property
    def equity(self) -> float:
        """Current account equity."""
        return self.initial_capital + self.realized_pnl

    def record_pnl(self, pnl: float) -> None:
        """Record realized profit or loss."""

        self.realized_pnl += pnl

        current_equity = self.equity

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = self.peak_equity - current_equity

        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
