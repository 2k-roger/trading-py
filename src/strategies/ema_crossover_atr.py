"""EMA Crossover entry with ATR-based trailing stop.

Strategy for Binance perpetual futures:
  Entry: EMA_short crosses above EMA_long → enter long.
  Stop:  Initial stop = entry_price − ATR × multiplier.
         Trailing stop ratchets up as price makes new highs.
  Exit:  Price closes below the trailing stop, or position is liquidated.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..models import TradeSetup, Position


@register('ema_crossover_atr')
class EMACrossoverATR(Strategy):
    """EMA-crossover trend-following with ATR trailing stop."""

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.ema_short = self.params.get('ema_short', 9)
        self.ema_long = self.params.get('ema_long', 21)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_multiplier = self.params.get('atr_multiplier', 3.0)

    # ------------------------------------------------------------------
    # Strategy API
    # ------------------------------------------------------------------

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA, ATR and crossover columns."""
        df = df.copy()

        # EMAs
        df['ema_short'] = df['close'].ewm(span=self.ema_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long, adjust=False).mean()

        # ATR (Wilder's smoothing via ewm with alpha=1/period)
        prev_close = df['close'].shift(1)
        tr = pd.concat(
            [
                df['high'] - df['low'],
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df['atr'] = tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()

        # Crossover detection: EMA_short crosses above EMA_long
        # Direct comparison avoids NaN dtype issues from shift
        above = df['ema_short'] > df['ema_long']
        above_prev = (df['ema_short'].shift(1) > df['ema_long'].shift(1)).fillna(False)
        df['entry_signal'] = (above & ~above_prev).astype(int)

        return df

    def on_bar(
        self,
        df: pd.DataFrame,
        idx: int,
        position: Optional[Position],
    ) -> TradeSetup:
        row = df.iloc[idx]

        if position is not None:
            # Already in a position — exits are handled by the engine
            return TradeSetup(action='none')

        # Flat — check for entry
        if row.get('entry_signal', 0) == 1:
            atr = row.get('atr', np.nan)
            if np.isnan(atr) or atr <= 0:
                return TradeSetup(action='none')
            initial_stop = row['close'] - self.atr_multiplier * atr
            return TradeSetup(
                action='enter_long',
                stop_loss=initial_stop,
            )

        return TradeSetup(action='none')

    def get_trailing_stop(self, position: Position, bar: pd.Series) -> float:
        """ATR trailing stop — only moves up (ratchets)."""
        if position.type != 'long':
            return position.stop_loss

        atr = bar.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return position.stop_loss

        # New potential stop = close − ATR × multiplier
        new_stop = bar['close'] - self.atr_multiplier * atr

        # Ratchet: only tighten (move up for long)
        return max(position.stop_loss, new_stop)
