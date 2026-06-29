"""Performance-metric computation from equity curve and trade list."""

import numpy as np
import pandas as pd

from ..models import Metrics, Trade

# Re-use the same module-level constants across calls
_RISK_FREE = 0.05  # 5 % annual


class MetricsCalculator:
    """Static methods to compute all backtest performance metrics."""

    @classmethod
    def compute(
        cls,
        equity_curve: pd.DataFrame,
        trades: list[Trade],
        initial_capital: float,
        risk_free_rate: float = _RISK_FREE,
    ) -> Metrics:
        """Compute full Metrics from an equity curve and trade list."""
        equity = equity_curve['equity'].values

        # Compute returns safely: avoid div-by-zero when equity ≈ 0
        with np.errstate(divide='ignore', invalid='ignore'):
            raw = np.diff(equity) / equity[:-1]
        returns = pd.Series(np.where(np.isfinite(raw), raw, 0.0), dtype=float)

        bars_per_year = cls._bars_per_year(equity_curve)
        n_bars = len(equity_curve)

        # ---- Return metrics ----
        final_equity = equity[-1]
        total_return_pct = (final_equity / initial_capital - 1.0) * 100.0

        cagr = (
            ((final_equity / initial_capital) ** (bars_per_year / n_bars) - 1.0)
            * 100.0
            if n_bars > 0 and initial_capital > 0
            else 0.0
        )

        # ---- Volatility ----
        ann_vol = returns.std() * np.sqrt(bars_per_year) * 100.0 if len(returns) > 1 else 0.0

        # ---- Sharpe ----
        excess = returns - risk_free_rate / bars_per_year
        sharpe = (
            np.sqrt(bars_per_year) * excess.mean() / returns.std()
            if len(returns) > 1 and returns.std() > 0
            else 0.0
        )

        # ---- Sortino ----
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(bars_per_year) if len(downside) > 1 else 1e-10
        sortino = (
            (returns.mean() - risk_free_rate / bars_per_year) * bars_per_year / downside_std
            if downside_std > 0
            else 0.0
        )

        # ---- Drawdown ----
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd_pct = float(np.min(drawdown) * 100.0)

        # Attach drawdown series to equity_curve for plotting
        equity_curve['drawdown'] = drawdown * 100.0

        max_dd_duration = cls._max_drawdown_duration(drawdown)

        # ---- Calmar ----
        calmar = (cagr / 100.0) / (abs(max_dd_pct) / 100.0) if max_dd_pct != 0 else 0.0

        # ---- Trade statistics ----
        total_trades = len(trades)
        winning = [t for t in trades if t.pnl_pct > 0]
        losing = [t for t in trades if t.pnl_pct <= 0]
        liquidations = sum(1 for t in trades if t.exit_reason == 'liquidation')

        win_rate = len(winning) / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(t.pnl_abs for t in winning)
        gross_loss = abs(sum(t.pnl_abs for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0.0
        avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0.0

        avg_holding = np.mean([t.holding_period for t in trades]) if trades else 0.0

        expectancy = (
            (win_rate * avg_win) - ((1.0 - win_rate) * abs(avg_loss))
            if total_trades > 0
            else 0.0
        )

        return_over_max_dd = (
            abs(total_return_pct / max_dd_pct) if max_dd_pct != 0 else float('inf')
        )

        return Metrics(
            total_return_pct=round(total_return_pct, 2),
            cagr=round(cagr, 2),
            volatility_pct=round(ann_vol, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            max_drawdown_duration=max_dd_duration,
            win_rate=round(win_rate, 4),
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            liquidations=liquidations,
            profit_factor=round(profit_factor, 2),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            avg_holding_period=round(avg_holding, 1),
            expectancy=round(expectancy, 2),
            return_over_max_dd=round(return_over_max_dd, 2),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_per_year(eq: pd.DataFrame) -> int:
        """Estimate bars-per-year from the index frequency."""
        if len(eq) < 2:
            return 365 * 24  # assume hourly
        delta = (eq.index[-1] - eq.index[0]).total_seconds()
        if delta <= 0:
            return 365 * 24
        return int(365.25 * 24 * 3600 / delta * len(eq))

    @staticmethod
    def _max_drawdown_duration(drawdown: np.ndarray) -> int:
        """Longest consecutive period (in bars) from peak to full recovery."""
        peak_idx = 0
        max_dur = 0
        in_dd = False
        current_dur = 0

        for i, dd in enumerate(drawdown):
            if dd < 0:
                if not in_dd:
                    peak_idx = i
                    in_dd = True
                current_dur = i - peak_idx
            else:
                in_dd = False
                max_dur = max(max_dur, current_dur)
                current_dur = 0

        return max(max_dur, current_dur)
