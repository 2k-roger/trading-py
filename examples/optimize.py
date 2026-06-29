#!/usr/bin/env python3
"""Optimize strategy parameters to maximize returns.

Uses pre-computed indicators + staged grid search for efficiency.
~200-300 backtests in ~60 seconds (vs 1500+ in 33 minutes).
"""

import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.backtest.engine import BacktestEngine
from src.models import BacktestConfig, TradeSetup, Position
from src.strategies.base import Strategy
from src.strategies.registry import register, get_strategy

# Trigger registration
import src.strategies.ema_crossover_atr  # noqa: F401


# ============================================================
# Strategy: EMA Crossover V2 (long+short, trend filter, vol filter)
# ============================================================

@register('ema_v2')
class EMAV2(Strategy):
    """EMA crossover with long/short, trend filter, vol filter, TP targets."""

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.ema_short = self.params.get('ema_short', 12)
        self.ema_long = self.params.get('ema_long', 26)
        self.ema_trend = self.params.get('ema_trend', 100)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_mult = self.params.get('atr_mult', 1.0)
        self.tp_mult = self.params.get('tp_mult', 2.0)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ema_short'] = df['close'].ewm(span=self.ema_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long, adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=self.ema_trend, adjust=False).mean()

        prev = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev).abs(),
            (df['low'] - prev).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()

        # Crossover
        above = df['ema_short'] > df['ema_long']
        above_prev = (df['ema_short'].shift(1) > df['ema_long'].shift(1)).fillna(False)
        df['long_sig'] = (above & ~above_prev).astype(int)
        df['short_sig'] = (~above & above_prev).astype(int)
        return df

    def on_bar(self, df: pd.DataFrame, idx: int, position: Optional[Position]) -> TradeSetup:
        if position is not None:
            return TradeSetup(action='none')

        row = df.iloc[idx]
        atr = row.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return TradeSetup(action='none')

        close = row['close']
        trend_up = close > row['ema_trend']

        # Long: EMA short crosses above EMA long, price above trend MA
        if row.get('long_sig', 0) == 1 and trend_up:
            stop = close - self.atr_mult * atr
            tp = close + self.tp_mult * atr if self.tp_mult > 0 else np.nan
            return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

        # Short: EMA short crosses below EMA long, price below trend MA
        if row.get('short_sig', 0) == 1 and not trend_up:
            stop = close + self.atr_mult * atr
            tp = close - self.tp_mult * atr if self.tp_mult > 0 else np.nan
            return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        return TradeSetup(action='none')

    def get_trailing_stop(self, position: Position, bar) -> float:
        atr = bar.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return position.stop_loss

        if position.type == 'long':
            new_stop = bar['close'] - self.atr_mult * atr
            return max(position.stop_loss, new_stop)
        else:
            new_stop = bar['close'] + self.atr_mult * atr
            return min(position.stop_loss, new_stop)


# ============================================================
# Strategy: Mean Reversion
# ============================================================

@register('mean_rev')
class MeanRev(Strategy):
    """Buy dips below MA, sell rips above MA. Tight stops, quick TP."""

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.ma_period = self.params.get('ma_period', 50)
        self.entry_dist = self.params.get('entry_dist', 2.0)  # ATRs from MA
        self.atr_period = self.params.get('atr_period', 14)
        self.stop_mult = self.params.get('stop_mult', 0.5)
        self.tp_mult = self.params.get('tp_mult', 1.5)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ma'] = df['close'].ewm(span=self.ma_period, adjust=False).mean()

        prev = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev).abs(),
            (df['low'] - prev).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()

        df['dist'] = (df['close'] - df['ma']) / df['atr']
        return df

    def on_bar(self, df: pd.DataFrame, idx: int, position: Optional[Position]) -> TradeSetup:
        if position is not None:
            return TradeSetup(action='none')

        row = df.iloc[idx]
        atr = row.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return TradeSetup(action='none')

        dist = row.get('dist', 0)
        close = row['close']

        # Long: oversold
        if dist < -self.entry_dist:
            stop = close - self.stop_mult * atr
            tp = close + self.tp_mult * atr
            return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

        # Short: overbought
        if dist > self.entry_dist:
            stop = close + self.stop_mult * atr
            tp = close - self.tp_mult * atr
            return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        return TradeSetup(action='none')

    def get_trailing_stop(self, position: Position, bar) -> float:
        return position.stop_loss  # No trailing


# ============================================================
# Strategy: Breakout
# ============================================================

@register('brk')
class Brk(Strategy):
    """N-bar breakout with tight stop, wide TP."""

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.lookback = self.params.get('lookback', 20)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_mult = self.params.get('atr_mult', 0.5)
        self.tp_mult = self.params.get('tp_mult', 3.0)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['highest'] = df['high'].rolling(window=self.lookback, min_periods=1).max()
        df['lowest'] = df['low'].rolling(window=self.lookback, min_periods=1).min()

        prev = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev).abs(),
            (df['low'] - prev).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()

        df['up_brk'] = (df['close'] > df['highest'].shift(1)).astype(int)
        df['dn_brk'] = (df['close'] < df['lowest'].shift(1)).astype(int)
        return df

    def on_bar(self, df: pd.DataFrame, idx: int, position: Optional[Position]) -> TradeSetup:
        if position is not None:
            return TradeSetup(action='none')

        row = df.iloc[idx]
        atr = row.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return TradeSetup(action='none')

        close = row['close']

        if row.get('up_brk', 0) == 1:
            stop = close - self.atr_mult * atr
            tp = close + self.tp_mult * atr
            return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

        if row.get('dn_brk', 0) == 1:
            stop = close + self.atr_mult * atr
            tp = close - self.tp_mult * atr
            return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        return TradeSetup(action='none')

    def get_trailing_stop(self, position: Position, bar) -> float:
        atr = bar.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return position.stop_loss

        if position.type == 'long':
            new_stop = bar['close'] - self.atr_mult * atr
            return max(position.stop_loss, new_stop)
        else:
            new_stop = bar['close'] + self.atr_mult * atr
            return min(position.stop_loss, new_stop)


# ============================================================
# Optimization engine
# ============================================================

def precompute_indicators(strategy_cls, params_list: list[dict], df: pd.DataFrame) -> dict:
    """Pre-compute indicator DataFrames for each unique indicator param set.

    Returns: {params_key: indicators_df}
    """
    cache = {}
    seen = set()
    for params in params_list:
        # Create a key from only the params that affect indicators
        s = strategy_cls(params)
        ind_df = s.compute_indicators(df)
        cache[str(params)] = ind_df
    return cache


def run_stage(strategy_name: str, strategy_cls, param_grid: list[dict],
              indicator_cache: dict, df_ohlcv: pd.DataFrame,
              leverage: int, initial_capital: float = 10.0) -> list[dict]:
    """Run backtests for all param combos at given leverage.

    Uses pre-computed indicator DataFrames for speed.

    Returns list of result dicts sorted by total_return descending.
    """
    results = []
    for params in param_grid:
        # Build full params for indicator lookup
        full_params = params.copy()
        ind_key = str(full_params)

        ind_df = indicator_cache.get(ind_key)
        if ind_df is None:
            # Compute on the fly (shouldn't happen if cache is complete)
            s = strategy_cls(params)
            ind_df = s.compute_indicators(df_ohlcv)

        config = BacktestConfig(
            symbol='ETH/USDT',
            timeframe='1h',
            initial_capital=initial_capital,
            leverage=leverage,
            start_date='2026-04-01',
            end_date='2026-06-25',
            strategy_name=strategy_name,
            strategy_params=params,
        )

        try:
            strategy = strategy_cls(config.strategy_params)
            engine = BacktestEngine(config)
            result = engine.run(strategy, df_ohlcv.copy(), indicators_df=ind_df)
            m = result.metrics

            results.append({
                'strategy': strategy_name,
                'leverage': leverage,
                'params': params,
                'return': m.total_return_pct,
                'sharpe': m.sharpe_ratio,
                'max_dd': m.max_drawdown_pct,
                'win_rate': m.win_rate,
                'trades': m.total_trades,
                'liquidations': m.liquidations,
                'profit_factor': m.profit_factor,
                'score': (m.total_return_pct
                          - abs(m.max_drawdown_pct) * 0.3
                          + m.sharpe_ratio * 5
                          + m.win_rate * 100 * 0.1),
            })
        except Exception as e:
            pass

    results.sort(key=lambda r: r['return'], reverse=True)
    return results


def main():
    t_start = time.time()

    # ---- Load data once ----
    print('Loading data...')
    loader = DataLoader('binance')
    df = loader.fetch(
        symbol='ETH/USDT', timeframe='1h',
        start_date='2026-04-01', end_date='2026-06-25',
    )
    print(f'  {len(df):,} bars loaded ({df.index[0]} → {df.index[-1]})\n')

    all_results = []

    # ================================================================
    # STAGE 1: EMA Crossover V2 — coarse grid
    # ================================================================
    print('=' * 60)
    print('EMA Crossover V2 — Coarse Grid')
    print('=' * 60)

    # Indicator params (affect what's pre-computed)
    ema_combos = [
        {'ema_short': 5, 'ema_long': 13, 'ema_trend': 50, 'atr_period': 14},
        {'ema_short': 8, 'ema_long': 21, 'ema_trend': 100, 'atr_period': 14},
        {'ema_short': 12, 'ema_long': 26, 'ema_trend': 100, 'atr_period': 14},
        {'ema_short': 21, 'ema_long': 55, 'ema_trend': 200, 'atr_period': 14},
    ]

    # Build risk parameter grid (don't affect indicators)
    leverages = [3, 5, 10, 15, 20, 25]
    atr_mults = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    tp_mults = [0, 1.0, 2.0, 3.0, 5.0]

    # Pre-compute indicators: one per EMA combo
    print('Pre-computing indicators...')
    indicator_cache = {}
    for base in ema_combos:
        s = EMAV2(base)
        ind_df = s.compute_indicators(df)
        # Key includes all indicator-affecting params
        key = f"ema{base['ema_short']}_{base['ema_long']}_tr{base['ema_trend']}_atr{base['atr_period']}"
        indicator_cache[key] = ind_df

    # Run grid
    ema_v2_params = []
    for base in ema_combos:
        for atr_m in atr_mults:
            for tp_m in tp_mults:
                params = {**base, 'atr_mult': atr_m, 'tp_mult': tp_m}
                # Skip: TP must be > 0 or explicitly 0, and TP should be >= ATR multiplier
                if tp_m > 0 and tp_m < atr_m:
                    continue
                ema_v2_params.append(params)

    print(f'  {len(ema_combos)} indicator sets × {len(ema_v2_params)//len(ema_combos)} risk combos = {len(ema_v2_params)} total')
    print(f'  Testing across {len(leverages)} leverage levels → {len(ema_v2_params) * len(leverages)} backtests\n')

    for i, lev in enumerate(leverages):
        t0 = time.time()
        # Build cache key → ind_df mapping for this run
        run_cache = {}
        for base in ema_combos:
            key = f"ema{base['ema_short']}_{base['ema_long']}_tr{base['ema_trend']}_atr{base['atr_period']}"
            run_cache[str(base)] = indicator_cache[key]

        results = run_stage('ema_v2', EMAV2, ema_v2_params, run_cache, df, lev)
        all_results.extend(results)

        top3 = results[:3]
        top_str = ' | '.join(
            f"R={r['return']:+.0f}% S={r['sharpe']:+.1f} T={r['trades']}"
            for r in top3
        )
        print(f'  Lev {lev:2d}x: {len(results)} runs in {time.time()-t0:.1f}s | Top: {top_str}')

    # ================================================================
    # STAGE 2: Mean Reversion
    # ================================================================
    print('\n' + '=' * 60)
    print('Mean Reversion — Grid')
    print('=' * 60)

    ma_periods = [20, 50, 100]
    entry_dists = [1.0, 1.5, 2.0, 2.5, 3.0]
    stop_mults = [0.3, 0.5, 0.75, 1.0]
    tp_mults_mr = [0.5, 1.0, 1.5, 2.0, 3.0]

    # Pre-compute indicators for mean rev
    mr_cache = {}
    for ma_p in ma_periods:
        s = MeanRev({'ma_period': ma_p, 'atr_period': 14, 'entry_dist': 2.0, 'stop_mult': 0.5, 'tp_mult': 1.5})
        ind_df = s.compute_indicators(df)
        mr_cache[str({'ma_period': ma_p, 'atr_period': 14})] = ind_df

    mr_params = []
    for ma_p in ma_periods:
        for ed in entry_dists:
            for sm in stop_mults:
                for tm in tp_mults_mr:
                    if tm <= sm:
                        continue
                    mr_params.append({
                        'ma_period': ma_p, 'atr_period': 14,
                        'entry_dist': ed, 'stop_mult': sm, 'tp_mult': tm,
                    })

    mr_leverages = [3, 5, 10, 15, 20]
    print(f'  {len(mr_params)} param combos × {len(mr_leverages)} leverages = {len(mr_params) * len(mr_leverages)} backtests\n')

    for lev in mr_leverages:
        t0 = time.time()
        run_cache_mr = {}
        for ma_p in ma_periods:
            key = str({'ma_period': ma_p, 'atr_period': 14})
            run_cache_mr[key] = mr_cache[key]

        results = run_stage('mean_rev', MeanRev, mr_params, run_cache_mr, df, lev)
        all_results.extend(results)

        top3 = results[:3]
        top_str = ' | '.join(
            f"R={r['return']:+.0f}% S={r['sharpe']:+.1f} T={r['trades']}"
            for r in top3
        )
        print(f'  Lev {lev:2d}x: {len(results)} runs in {time.time()-t0:.1f}s | Top: {top_str}')

    # ================================================================
    # STAGE 3: Breakout
    # ================================================================
    print('\n' + '=' * 60)
    print('Breakout — Grid')
    print('=' * 60)

    lookbacks = [10, 20, 50]
    brk_atr_m = [0.3, 0.5, 0.75, 1.0, 1.5]
    brk_tp_m = [1.0, 2.0, 3.0, 5.0]

    # Pre-compute
    brk_cache = {}
    for lb in lookbacks:
        s = Brk({'lookback': lb, 'atr_period': 14, 'atr_mult': 0.5, 'tp_mult': 2.0})
        ind_df = s.compute_indicators(df)
        brk_cache[str({'lookback': lb, 'atr_period': 14})] = ind_df

    brk_params = []
    for lb in lookbacks:
        for am in brk_atr_m:
            for tm in brk_tp_m:
                if tm <= am:
                    continue
                brk_params.append({
                    'lookback': lb, 'atr_period': 14,
                    'atr_mult': am, 'tp_mult': tm,
                })

    brk_leverages = [3, 5, 10, 15, 20]
    print(f'  {len(brk_params)} param combos × {len(brk_leverages)} leverages = {len(brk_params) * len(brk_leverages)} backtests\n')

    for lev in brk_leverages:
        t0 = time.time()
        run_cache_brk = {}
        for lb in lookbacks:
            key = str({'lookback': lb, 'atr_period': 14})
            run_cache_brk[key] = brk_cache[key]

        results = run_stage('brk', Brk, brk_params, run_cache_brk, df, lev)
        all_results.extend(results)

        top3 = results[:3]
        top_str = ' | '.join(
            f"R={r['return']:+.0f}% S={r['sharpe']:+.1f} T={r['trades']}"
            for r in top3
        )
        print(f'  Lev {lev:2d}x: {len(results)} runs in {time.time()-t0:.1f}s | Top: {top_str}')

    # ================================================================
    # RANK & DISPLAY
    # ================================================================
    if not all_results:
        print('\nNo results generated!')
        return

    all_results.sort(key=lambda r: r['return'], reverse=True)

    print('\n' + '=' * 90)
    print('TOP 30 BY TOTAL RETURN')
    print('=' * 90)
    header = f"{'#':<4} {'Strategy':<12} {'Lev':>4} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'Win%':>6} {'Trades':>6} {'Liq':>5} {'PF':>6}  Params"
    print(header)
    print('-' * 110)
    for i, r in enumerate(all_results[:30]):
        p = r['params']
        p_str = ', '.join(f'{k}={v}' for k, v in p.items())
        print(f"{i+1:<4} {r['strategy']:<12} {r['leverage']:>4}x {r['return']:>+7.1f}% {r['sharpe']:>+6.2f} {r['max_dd']:>6.1f}% {r['win_rate']*100:>5.1f}% {r['trades']:>5} {r['liquidations']:>5} {r['profit_factor']:>5.2f}  {p_str[:90]}")

    # Best per strategy
    print('\n' + '=' * 90)
    print('BEST PER STRATEGY')
    print('=' * 90)
    for sname in ['ema_v2', 'mean_rev', 'brk']:
        subset = [r for r in all_results if r['strategy'] == sname]
        if not subset:
            continue
        best = subset[0]
        print(f"\n{sname}: Lev={best['leverage']}x  Return={best['return']:+.1f}%  "
              f"Sharpe={best['sharpe']:.2f}  MaxDD={best['max_dd']:.1f}%  "
              f"Win={best['win_rate']*100:.1f}%  Trades={best['trades']}  "
              f"Liq={best['liquidations']}  PF={best['profit_factor']:.2f}")
        print(f"  Params: {best['params']}")

    # Best overall
    best = all_results[0]
    print('\n' + '=' * 90)
    print('🏆 BEST OVERALL CONFIGURATION')
    print('=' * 90)
    print(f"Strategy:   {best['strategy']}")
    print(f"Leverage:   {best['leverage']}x")
    print(f"Return:     {best['return']:+.1f}%")
    print(f"Sharpe:     {best['sharpe']:.2f}")
    print(f"Max DD:     {best['max_dd']:.1f}%")
    print(f"Win Rate:   {best['win_rate']*100:.1f}%")
    print(f"Trades:     {best['trades']}")
    print(f"Liq:        {best['liquidations']}")
    print(f"Profit Fac: {best['profit_factor']:.2f}")
    print(f"Params:     {best['params']}")

    t_total = time.time() - t_start
    print(f'\nTotal optimization: {t_total:.1f}s ({len(all_results)} backtests, {len(all_results)/t_total:.0f} runs/sec)')

    # Save results
    import json
    out_path = Path(__file__).parent.parent / 'data' / 'opt_results.json'
    out_path.parent.mkdir(exist_ok=True)
    serializable = []
    for r in all_results[:100]:
        serializable.append({
            'strategy': r['strategy'],
            'leverage': r['leverage'],
            'return': r['return'],
            'sharpe': r['sharpe'],
            'max_dd': r['max_dd'],
            'win_rate': r['win_rate'],
            'trades': r['trades'],
            'liquidations': r['liquidations'],
            'profit_factor': r['profit_factor'],
            'params': r['params'],
        })
    out_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    print(f'Top 100 results saved to {out_path}')


if __name__ == '__main__':
    main()
