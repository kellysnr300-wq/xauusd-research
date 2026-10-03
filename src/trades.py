from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Position:
    """Represents an open position."""

    side: str
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Trade:
    """Represents a completed trade."""

    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    exit_reason: str
