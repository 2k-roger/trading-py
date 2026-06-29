"""Bar-by-bar backtesting engine with leverage and liquidation support.

Optimized with numpy-backed bars for ~5x faster loop execution.
"""

import numpy as np
import pandas as pd

from ..models import (
    BacktestConfig,
    BacktestResult,
    Position,
    Trade,
    TradeSetup,
)
from ..risk.manager import RiskManager
from ..strategies.base import Strategy
from ..metrics.calculator import MetricsCalculator


class FastBar:
    """Lightweight proxy for a single bar backed by numpy arrays.

    Drop-in replacement for pd.Series in the hot loop — supports
    __getitem__ and .get() with identical semantics but zero
    pandas overhead.
    """
    __slots__ = ('_arrays', '_idx')

    def __init__(self, arrays: dict, idx: int):
        self._arrays = arrays
        self._idx = idx

    def __getitem__(self, key):
        return self._arrays[key][self._idx]

    def get(self, key, default=None):
        arr = self._arrays.get(key)
        if arr is None:
            return default
        val = arr[self._idx]
        if np.isnan(val):
            return default
        return val

    # Convenience properties for the engine's internal use
    @property
    def open(self) -> float:
        return self._arrays['open'][self._idx]

    @property
    def high(self) -> float:
        return self._arrays['high'][self._idx]

    @property
    def low(self) -> float:
        return self._arrays['low'][self._idx]

    @property
    def close(self) -> float:
        return self._arrays['close'][self._idx]


class BacktestEngine:
    """Simulates leveraged futures trading bar-by-bar.

    Exit-check order (each bar):
      1. Liquidation — did price cross the liquidation threshold?
      2. Stop-loss — did price hit the stop?
      3. Take-profit — did price hit the target?
      4. Trailing-stop update — ratchet the stop if applicable.

    Equity = cash + unrealised PnL of any open position.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.risk = RiskManager(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, strategy: Strategy, data: pd.DataFrame,
            indicators_df: pd.DataFrame | None = None) -> BacktestResult:
        """Execute the backtest.

        Args:
            strategy: Instantiated Strategy object.
            data: OHLCV DataFrame with datetime index.
            indicators_df: Pre-computed indicators (optional). If None,
                strategy.compute_indicators() is called.

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        # ---- 1. Compute / use indicators ----
        if indicators_df is not None:
            df = indicators_df
        else:
            df = strategy.compute_indicators(data)

        n_bars = len(df)

        # ---- 2. Pre-convert to numpy arrays (the key optimization) ----
        arrays = {col: df[col].values for col in df.columns}
        times = df.index.values  # numpy datetime64 array

        # ---- 3. Initialise state ----
        cash = self.config.initial_capital
        position: Position | None = None
        trades: list[Trade] = []

        # Pre-allocate equity record arrays
        eq_timestamps = [None] * n_bars
        eq_values = np.empty(n_bars, dtype=np.float64)
        eq_cash = np.empty(n_bars, dtype=np.float64)
        eq_in_pos = np.empty(n_bars, dtype=np.bool_)

        # ---- 4. Bar-by-bar simulation (numpy-backed) ----
        for idx in range(n_bars):
            # Fast path: use numpy array directly
            bar_high = arrays['high'][idx]
            bar_low = arrays['low'][idx]
            bar_close = arrays['close'][idx]
            bar_time = times[idx]

            # --- 4a. Check exits on open position ---
            if position is not None:
                exit_price, exit_reason = self._check_exits_fast(
                    position, bar_high, bar_low
                )

                if exit_price is not None:
                    trade = self._close_position(
                        position, exit_price, exit_reason, bar_time, idx
                    )
                    trades.append(trade)

                    # Release margin + PnL - fees (cap at 0: isolated margin)
                    released = position.margin + trade.pnl_abs - trade.exit_fee
                    cash += max(0.0, released)
                    position = None

            # --- 4b. Check entry if flat ---
            if position is None and cash > 0:
                setup = strategy.on_bar(df, idx, position)
                if setup.action in ('enter_long', 'enter_short'):
                    position = self._open_position_fast(
                        setup, arrays, idx, cash, bar_time
                    )
                    if position is not None:
                        # Deduct entry fee + lock margin
                        entry_fee = self.risk.trade_fee(position.position_value)
                        cash -= entry_fee + position.margin

            # --- 4c. Update trailing stop ---
            if position is not None:
                bar_proxy = FastBar(arrays, idx)
                new_stop = strategy.get_trailing_stop(position, bar_proxy)
                if position.type == 'long' and new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                elif position.type == 'short' and new_stop < position.stop_loss:
                    position.stop_loss = new_stop

                # Update high/low watermarks
                if bar_high > position.highest_price:
                    position.highest_price = bar_high
                if bar_low < position.lowest_price:
                    position.lowest_price = bar_low

            # --- 4d. Record equity snapshot ---
            equity = cash
            if position is not None:
                pnl = (bar_close - position.entry_price) * position.size
                if position.type == 'short':
                    pnl = (position.entry_price - bar_close) * position.size
                equity += pnl

            eq_timestamps[idx] = bar_time
            eq_values[idx] = equity
            eq_cash[idx] = cash
            eq_in_pos[idx] = position is not None

        # ---- 5. Force-close any open position at final bar ----
        if position is not None:
            final_close = arrays['close'][-1]
            final_time = times[-1]
            trade = self._close_position(
                position, final_close, 'end_of_data', final_time, n_bars - 1
            )
            trades.append(trade)
            released = position.margin + trade.pnl_abs - trade.exit_fee
            cash += max(0.0, released)
            eq_values[-1] = cash
            eq_in_pos[-1] = False

        # ---- 6. Build result ----
        eq_df = pd.DataFrame({
            'timestamp': eq_timestamps,
            'equity': eq_values,
            'cash': eq_cash,
            'in_position': eq_in_pos,
        })
        eq_df.set_index('timestamp', inplace=True)

        metrics = MetricsCalculator.compute(
            eq_df, trades, self.config.initial_capital
        )

        return BacktestResult(
            config=self.config,
            df=df,
            equity_curve=eq_df,
            trades=trades,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Exit logic (optimized: takes scalars instead of Series)
    # ------------------------------------------------------------------

    def _check_exits_fast(
        self, position: Position, bar_high: float, bar_low: float
    ) -> tuple[float | None, str | None]:
        """Check all exit conditions in priority order using raw floats.

        Returns (exit_price, reason) or (None, None).
        """
        if position.type == 'long':
            # 1. Liquidation
            if bar_low <= position.liquidation_price:
                return position.liquidation_price, 'liquidation'

            # 2. Stop-loss (distinguish initial vs trailing)
            if bar_low <= position.stop_loss:
                if position.stop_loss > position.initial_stop_loss:
                    return self._slipped(position.stop_loss, 'exit'), 'trailing_stop'
                return self._slipped(position.stop_loss, 'exit'), 'stop_loss'

            # 3. Take-profit
            if not np.isnan(position.take_profit) and bar_high >= position.take_profit:
                return self._slipped(position.take_profit, 'exit'), 'take_profit'

            return None, None

        else:  # short
            # 1. Liquidation
            if bar_high >= position.liquidation_price:
                return position.liquidation_price, 'liquidation'

            # 2. Stop-loss (distinguish initial vs trailing)
            if bar_high >= position.stop_loss:
                if position.stop_loss < position.initial_stop_loss:
                    return self._slipped(position.stop_loss, 'exit'), 'trailing_stop'
                return self._slipped(position.stop_loss, 'exit'), 'stop_loss'

            # 3. Take-profit
            if not np.isnan(position.take_profit) and bar_low <= position.take_profit:
                return self._slipped(position.take_profit, 'exit'), 'take_profit'

            return None, None

    # Kept for backward compatibility
    def _check_exits(
        self, position: Position, bar: pd.Series
    ) -> tuple[float | None, str | None]:
        return self._check_exits_fast(position, bar['high'], bar['low'])

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def _open_position_fast(
        self,
        setup: TradeSetup,
        arrays: dict,
        idx: int,
        cash: float,
        bar_time,
    ) -> Position | None:
        """Open a new leveraged position using numpy arrays."""
        entry_price = float(arrays['close'][idx])

        # Position size based on risk model
        size = self.risk.compute_position_size(
            cash, entry_price, setup.stop_loss, setup.action.replace('enter_', '')
        )
        if size <= 0:
            return None

        position_value = entry_price * size
        margin = self.risk.required_margin(position_value, self.config.leverage)
        entry_fee = self.risk.trade_fee(position_value)

        # Can we afford margin + fee? Must leave room for entry_fee deduction.
        total_required = margin + entry_fee
        if total_required > cash:
            # Scale position value to fit: position_value / leverage + position_value * fee_pct ≤ cash
            max_position_value = cash / (1.0 / self.config.leverage + self.config.commission_pct)
            size = max_position_value / entry_price
            if size <= 0:
                return None
            position_value = entry_price * size
            margin = self.risk.required_margin(position_value, self.config.leverage)
            entry_fee = self.risk.trade_fee(position_value)
            # Safety: if still can't afford, return None
            if margin + entry_fee > cash * 1.001:  # tiny tolerance for float rounding
                return None

        side = 'long' if 'long' in setup.action else 'short'
        liq_price = self.risk.liquidation_price(
            entry_price, self.config.leverage, side
        )

        bar_high = float(arrays['high'][idx])
        bar_low = float(arrays['low'][idx])

        return Position(
            type=side,
            entry_time=bar_time,
            entry_price=entry_price,
            size=size,
            margin=margin,
            leverage=self.config.leverage,
            entry_idx=idx,
            stop_loss=setup.stop_loss or np.nan,
            initial_stop_loss=setup.stop_loss or np.nan,
            take_profit=setup.take_profit if setup.take_profit else np.nan,
            liquidation_price=liq_price,
            highest_price=bar_high,
            lowest_price=bar_low,
        )

    def _open_position(
        self, setup, bar, cash, bar_time, entry_idx
    ) -> Position | None:
        """Legacy wrapper — kept for compatibility."""
        arrays = {col: np.array([bar[col]]) for col in bar.index}
        return self._open_position_fast(setup, arrays, 0, cash, bar_time)

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        reason: str,
        exit_time,
        exit_idx: int,
    ) -> Trade:
        """Create a Trade record for a closed position."""
        if reason == 'liquidation':
            # At liquidation, the entire margin is lost
            pnl_abs = -position.margin
        else:
            pnl_abs = position.unrealized_pnl(exit_price)
            # Isolated margin: loss cannot exceed margin
            if pnl_abs < -position.margin:
                pnl_abs = -position.margin

        pnl_pct = (pnl_abs / position.margin * 100) if position.margin > 0 else 0.0
        exit_fee = self.risk.trade_fee(position.position_value)
        holding_period = exit_idx - position.entry_idx

        return Trade(
            type=position.type,
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            margin=position.margin,
            leverage=position.leverage,
            pnl_abs=pnl_abs,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            holding_period=holding_period,
            entry_fee=0.0,  # already deducted from cash at open
            exit_fee=exit_fee,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _slipped(self, price: float, direction: str) -> float:
        """Apply slippage to a fill price."""
        slip = self.config.slippage_pct
        if direction == 'entry':
            return price * (1.0 + slip)  # worse for buyer
        return price * (1.0 - slip)  # worse for seller
