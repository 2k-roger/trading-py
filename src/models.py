"""Shared data classes for the trading system.

All modules import from here. No circular dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """A single completed trade."""

    type: str  # 'long' | 'short'
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float  # contract size (units of base currency)
    margin: float  # margin locked for this position
    leverage: int
    pnl_abs: float  # absolute PnL in quote currency
    pnl_pct: float  # percentage return on margin
    exit_reason: str  # 'stop_loss' | 'take_profit' | 'trailing_stop' | 'signal' | 'liquidation' | 'end_of_data'
    holding_period: int  # bars held
    entry_fee: float = 0.0
    exit_fee: float = 0.0


@dataclass
class Position:
    """An open leveraged position being tracked by the engine."""

    type: str  # 'long' | 'short'
    entry_time: datetime
    entry_price: float
    size: float  # contract size (units of base currency)
    margin: float  # locked margin for this position
    leverage: int
    entry_idx: int = 0  # bar index when position was opened
    stop_loss: float = np.nan
    initial_stop_loss: float = np.nan  # stop at entry (to distinguish trailing vs initial)
    take_profit: float = np.nan
    liquidation_price: float = np.nan
    highest_price: float = 0.0  # highest price since entry (for long trailing)
    lowest_price: float = np.inf  # lowest price since entry (for short trailing)

    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized PnL based on current mark price."""
        if self.type == 'long':
            return (current_price - self.entry_price) * self.size
        return (self.entry_price - current_price) * self.size

    @property
    def position_value(self) -> float:
        """Notional value of the position."""
        return self.entry_price * self.size


@dataclass
class TradeSetup:
    """What the strategy wants to do at the current bar."""

    action: str  # 'enter_long' | 'enter_short' | 'exit' | 'none'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Metrics:
    """Computed performance metrics for a backtest."""

    total_return_pct: float
    cagr: float  # Compound Annual Growth Rate
    volatility_pct: float  # annualized
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration: int  # in bars
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    liquidations: int  # number of forced liquidations
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_holding_period: float  # in bars
    expectancy: float  # avg PnL per trade (in %)
    return_over_max_dd: float

    def display(self) -> str:
        """Format metrics as a multi-line string. Labels in Chinese."""
        lines = [
            "=" * 60,
            f"  {'指标':<26} {'数值':>12}",
            "-" * 42,
            f"  {'总收益率':<26} {self.total_return_pct:>+10.2f}%",
            f"  {'年化收益率 (CAGR)':<26} {self.cagr:>+10.2f}%",
            f"  {'年化波动率':<26} {self.volatility_pct:>10.2f}%",
            f"  {'夏普比率 (Sharpe)':<26} {self.sharpe_ratio:>10.2f}",
            f"  {'索提诺比率 (Sortino)':<26} {self.sortino_ratio:>10.2f}",
            f"  {'卡尔玛比率 (Calmar)':<26} {self.calmar_ratio:>10.2f}",
            f"  {'最大回撤':<26} {self.max_drawdown_pct:>10.2f}%",
            f"  {'最长回撤持续 (K线数)':<26} {self.max_drawdown_duration:>10}",
            f"  {'胜率':<26} {self.win_rate*100:>10.1f}%",
            f"  {'总交易次数':<26} {self.total_trades:>10}",
            f"  {'  盈利笔数':<26} {self.winning_trades:>10}",
            f"  {'  亏损笔数':<26} {self.losing_trades:>10}",
            f"  {'  爆仓笔数':<26} {self.liquidations:>10}",
            f"  {'盈亏比 (Profit Factor)':<26} {self.profit_factor:>10.2f}",
            f"  {'每笔期望收益':<26} {self.expectancy:>+10.2f}%",
            f"  {'平均盈利':<26} {self.avg_win_pct:>+10.2f}%",
            f"  {'平均亏损':<26} {self.avg_loss_pct:>+10.2f}%",
            f"  {'平均持仓 (K线数)':<26} {self.avg_holding_period:>10.1f}",
            f"  {'收益/最大回撤':<26} {self.return_over_max_dd:>10.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    # Data
    exchange: str = 'binance'
    symbol: str = 'ETH/USDT'
    timeframe: str = '1h'
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Capital & leverage
    initial_capital: float = 10.0  # USDT
    leverage: int = 100
    margin_mode: str = 'isolated'  # 'isolated' | 'cross'
    maintenance_margin_pct: float = 0.005  # 0.5% for ETH/USDT perpetual

    # Costs
    commission_pct: float = 0.0004  # 0.04% taker fee per trade
    slippage_pct: float = 0.0001  # 0.01% slippage

    # Risk
    position_sizing: str = 'fixed_risk'  # 'fixed_risk' | 'fixed_units' | 'percent_equity'
    risk_per_trade_pct: float = 0.01  # 1% of capital risked per trade

    # Strategy
    strategy_name: str = 'ema_crossover_atr'
    strategy_params: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete result of a backtest run."""

    config: BacktestConfig
    df: pd.DataFrame  # OHLCV + indicators
    equity_curve: pd.DataFrame  # columns: timestamp, equity, cash, in_position, drawdown
    trades: list  # list[Trade]
    metrics: Metrics

    def summary(self) -> str:
        """One-line summary."""
        eq = self.equity_curve
        return (
            f"{self.config.strategy_name} | {self.config.symbol} {self.config.timeframe} | "
            f"{eq.index[0].date()} → {eq.index[-1].date()} | "
            f"${eq['equity'].iloc[-1]:.2f} | "
            f"{self.metrics.total_return_pct:+.1f}% | "
            f"Sharpe {self.metrics.sharpe_ratio:.2f}"
        )
