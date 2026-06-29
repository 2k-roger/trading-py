"""Position sizing, liquidation-price calculation, and fee handling."""

import numpy as np

from ..models import BacktestConfig


class RiskManager:
    """Computes position sizes and liquidation prices for leveraged futures."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        side: str = 'long',
    ) -> float:
        """Return contract size (units of base currency).

        Respects the max position value = capital × leverage.
        """
        method = self.config.position_sizing
        max_position_value = capital * self.config.leverage

        if method == 'fixed_risk':
            risk_amount = capital * self.config.risk_per_trade_pct
            price_risk = abs(entry_price - stop_loss)
            if price_risk <= 0:
                return 0.0
            size = risk_amount / price_risk
        elif method == 'fixed_units':
            size = float(self.config.strategy_params.get('fixed_units', 1))
        elif method == 'percent_equity':
            size = (capital * self.config.risk_per_trade_pct) / entry_price
        else:
            raise ValueError(f"Unknown position sizing method: {method}")

        # Cap by max position value
        max_size = max_position_value / entry_price
        return min(size, max_size)

    def liquidation_price(
        self,
        entry_price: float,
        leverage: int,
        side: str = 'long',
    ) -> float:
        """Calculate the liquidation price for an isolated-margin position.

        Formula (Binance):
          long:  entry × (1 − 1/leverage + mmr)
          short: entry × (1 + 1/leverage − mmr)

        where mmr = maintenance margin rate.
        """
        mmr = self.config.maintenance_margin_pct
        inv_lev = 1.0 / leverage

        if side == 'long':
            return entry_price * (1.0 - inv_lev + mmr)
        else:
            return entry_price * (1.0 + inv_lev - mmr)

    def required_margin(self, position_value: float, leverage: int) -> float:
        """Initial margin needed to open a position."""
        return position_value / leverage

    def trade_fee(self, position_value: float) -> float:
        """Commission for one side of a trade."""
        return position_value * self.config.commission_pct
