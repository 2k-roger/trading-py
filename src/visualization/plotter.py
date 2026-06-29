"""Visualise backtest results with matplotlib."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..models import BacktestResult


class Plotter:
    """Static methods for plotting backtest results."""

    @staticmethod
    def summary(result: BacktestResult, save_path: Optional[str] = None):
        """Four-panel dashboard:
          1. Equity curve + buy & hold benchmark
          2. Drawdown
          3. Price with entry/exit markers
          4. Per-trade PnL bars + metrics box
        """
        eq = result.equity_curve
        df = result.df
        config = result.config
        trades = result.trades

        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

        # ---- Panel 1: Equity curve ----
        ax1 = axes[0]
        ax1.plot(
            eq.index, eq['equity'], linewidth=1.5, color='navy', label='Portfolio'
        )
        # Buy & hold benchmark
        bh = df['close'] / df['close'].iloc[0] * config.initial_capital
        ax1.plot(
            bh.index, bh.values, linewidth=1, color='gray', alpha=0.6,
            label='Buy & Hold',
        )
        ax1.axhline(
            y=config.initial_capital, color='black', linestyle='--', alpha=0.3,
        )
        ax1.fill_between(
            eq.index, config.initial_capital, eq['equity'],
            where=eq['equity'] >= config.initial_capital,
            color='green', alpha=0.1,
        )
        ax1.fill_between(
            eq.index, config.initial_capital, eq['equity'],
            where=eq['equity'] < config.initial_capital,
            color='red', alpha=0.1,
        )

        # Trade exit markers
        for t in trades:
            color = 'green' if t.pnl_abs > 0 else 'red'
            marker = 'x' if t.exit_reason == 'liquidation' else 'v'
            size = 60 if t.exit_reason == 'liquidation' else 30
            if t.exit_time in eq.index:
                y = eq.loc[t.exit_time, 'equity']
            else:
                y = eq['equity'].iloc[-1]
            ax1.scatter(t.exit_time, y, color=color, s=size, marker=marker, zorder=5)

        ax1.set_title(
            f'{config.strategy_name} | {config.symbol} {config.timeframe} | '
            f'{config.leverage}x | {config.initial_capital} USDT'
        )
        ax1.set_ylabel('Equity (USDT)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # ---- Panel 2: Drawdown ----
        ax2 = axes[1]
        ax2.fill_between(eq.index, 0, eq['drawdown'], color='red', alpha=0.3)
        ax2.plot(eq.index, eq['drawdown'], color='red', linewidth=1)
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # ---- Panel 3: Price + trades ----
        ax3 = axes[2]
        ax3.plot(
            df.index, df['close'], linewidth=1, color='black', alpha=0.7,
            label='Close',
        )
        # EMA lines if present
        if 'ema_short' in df.columns:
            ax3.plot(
                df.index, df['ema_short'], linewidth=0.8, alpha=0.5,
                label=f"EMA({config.strategy_params.get('ema_short', 9)})",
            )
        if 'ema_long' in df.columns:
            ax3.plot(
                df.index, df['ema_long'], linewidth=0.8, alpha=0.5,
                label=f"EMA({config.strategy_params.get('ema_long', 21)})",
            )

        for t in trades:
            ax3.scatter(
                t.entry_time, t.entry_price, color='blue', s=50, marker='^', zorder=5,
            )
            color = 'cyan' if t.exit_reason == 'liquidation' else (
                'green' if t.pnl_abs > 0 else 'red'
            )
            ax3.scatter(
                t.exit_time, t.exit_price, color=color, s=50, marker='v', zorder=5,
            )
            # Connect entry → exit
            ax3.plot(
                [t.entry_time, t.exit_time],
                [t.entry_price, t.exit_price],
                linewidth=0.6, color=color, alpha=0.4,
            )

        ax3.set_ylabel('Price (USDT)')
        ax3.legend(loc='upper left', fontsize='small')
        ax3.grid(True, alpha=0.3)

        # ---- Panel 4: PnL distribution ----
        ax4 = axes[3]
        pnls = [t.pnl_pct for t in trades]
        colors = [
            'cyan' if t.exit_reason == 'liquidation'
            else ('green' if p > 0 else 'red')
            for t, p in zip(trades, pnls)
        ]
        ax4.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=0.7)
        ax4.axhline(y=0, color='black', linewidth=0.5)
        ax4.set_xlabel('Trade #')
        ax4.set_ylabel('PnL (%)')
        ax4.grid(True, alpha=0.3)

        # Build metrics text box
        m = result.metrics
        text = (
            f"Return: {m.total_return_pct:+.2f}%  CAGR: {m.cagr:+.2f}%  "
            f"Sharpe: {m.sharpe_ratio:.2f}  MaxDD: {m.max_drawdown_pct:.1f}%  "
            f"Liq: {m.liquidations}\n"
            f"Win: {m.win_rate*100:.0f}%  Trades: {m.total_trades}  "
            f"PF: {m.profit_factor:.2f}  Exp: {m.expectancy:+.2f}%"
        )
        ax4.text(
            0.5, -0.45, text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        plt.show()

    @staticmethod
    def print_metrics(result: BacktestResult):
        """Print the metrics display string to stdout."""
        print(result.metrics.display())
