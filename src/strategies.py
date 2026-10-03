from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base interface for research strategies."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate strategy signals from market data."""
        raise NotImplementedError
