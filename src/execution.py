from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionModel:
    """Defines basic simulated order execution assumptions."""

    spread: float = 0.0
    slippage: float = 0.0

    def buy_price(self, market_price: float) -> float:
        """Simulated price for a long entry."""
        return market_price + (self.spread / 2) + self.slippage

    def sell_price(self, market_price: float) -> float:
        """Simulated price for a short entry."""
        return market_price - (self.spread / 2) - self.slippage

    def exit_long_price(self, market_price: float) -> float:
        """Simulated exit price for a long position."""
        return market_price - (self.spread / 2) - self.slippage

    def exit_short_price(self, market_price: float) -> float:
        """Simulated exit price for a short position."""
        return market_price + (self.spread / 2) + self.slippage
