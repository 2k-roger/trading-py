"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from ..models import TradeSetup, Position


class Strategy(ABC):
    """Base class for all trading strategies.

    Lifecycle during backtesting:
      1. compute_indicators(df) — called once on full dataset.
      2. on_bar(df, idx, position) — called for each bar in sequence.
      3. get_trailing_stop(position, bar) — called each bar when in position.

    Subclasses must implement ``compute_indicators`` and ``on_bar``.
    ``get_trailing_stop`` has a default no-op implementation.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}

    @abstractmethod
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to a *copy* of df.

        Called once before the bar-by-bar loop.  Must not mutate the
        original DataFrame.
        """
        ...

    @abstractmethod
    def on_bar(
        self,
        df: pd.DataFrame,
        idx: int,
        position: Optional[Position],
    ) -> TradeSetup:
        """Decision at the current bar.

        Args:
            df: Full DataFrame with indicator columns.
            idx: Current bar index.
            position: Currently open position, or None if flat.

        Returns:
            TradeSetup describing the desired action.
        """
        ...

    def get_trailing_stop(self, position: Position, bar: pd.Series) -> float:
        """Compute updated trailing-stop price.

        Override in subclasses that use trailing stops.  Return the
        *new* stop price (must be ≥ current for long, ≤ for short).
        Default: no trailing — returns current stop_loss unchanged.
        """
        return position.stop_loss
