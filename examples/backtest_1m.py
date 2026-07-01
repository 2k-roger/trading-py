#!/usr/bin/env python3
"""1m 波动突破策略回测 — 多参数对比。

用法:
    python examples/backtest_1m.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import ConfigLoader
from src.data.loader import DataLoader
from src.backtest.engine import BacktestEngine
from src.models import BacktestConfig

# 注册策略
import src.strategies.volatility_breakout_1m  # noqa: F401
from src.strategies.registry import get_strategy

_REASON_CN = {
    'stop_loss':     '初始止损',
    'take_profit':   '止盈',
    'trailing_stop': '移动止损',
    'liquidation':   '爆仓',
    'end_of_data':   '数据结束',
    'signal':        '信号反转',
}


def run_single(config_overrides: dict, strategy_params: dict, label: str = ''):
    """运行单次回测并打印结果。"""
    config = ConfigLoader.load(overrides=config_overrides)

    loader = DataLoader(config.exchange)
    df = loader.fetch(
        symbol=config.symbol,
        timeframe=config.timeframe,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    strategy_cls = get_strategy(config.strategy_name)
    strategy = strategy_cls(config.strategy_params)

    engine = BacktestEngine(config)
    result = engine.run(strategy, df)

    if label:
        print(f'\n{"="*70}')
        print(f'  {label}')
        print(f'{"="*70}')

    print(f'\n数据: {config.symbol} {config.timeframe} | '
          f'{len(df):,} 根K线 | {df.index[0]} → {df.index[-1]}')
    print(f'策略: {config.strategy_name} | '
          f'杠杆: {config.leverage}x | 本金: ${config.initial_capital}')

    print(result.metrics.display())

    final_equity = result.equity_curve['equity'].iloc[-1]
    pnl = final_equity - config.initial_capital
    print(f'初始资金: ${config.initial_capital:.2f}')
    print(f'最终权益: ${final_equity:.2f}')
    print(f'净盈亏:   ${pnl:+.2f} ({pnl/config.initial_capital*100:+.1f}%)')

    # 交易明细摘要
    if result.trades:
        print(f'\n{"序号":<5} {"入场":<20} {"出场":<20} '
              f'{"方向":<6} {"入场价":>10} {"出场价":>10} {"盈亏%":>8} {"出场原因":<12}')
        print('-' * 90)
        for i, t in enumerate(result.trades, 1):
            d = '做多' if t.type == 'long' else '做空'
            r = _REASON_CN.get(t.exit_reason, t.exit_reason)
            print(f'{i:<5} {str(t.entry_time):<20} {str(t.exit_time):<20} '
                  f'{d:<6} {t.entry_price:>10.2f} {t.exit_price:>10.2f} '
                  f'{t.pnl_pct:>+7.2f}% {r:<12}')
    else:
        print('\n⚠️  无交易信号')

    return result


def grid_search():
    """网格搜索最优参数组合。"""
    print('╔══════════════════════════════════════════════════╗')
    print('║ 1m 波动突破策略 — 参数网格搜索                    ║')
    print('╚══════════════════════════════════════════════════╝')

    # 加载数据一次
    loader = DataLoader('binance')
    df = loader.fetch(
        symbol='ETH/USDT', timeframe='1m',
        start_date='2026-06-24', end_date='2026-07-01',
        force_download=False,
    )
    print(f'\n数据: {len(df):,} 根 1m K线 ({df.index[0]} → {df.index[-1]})')

    # 参数网格
    param_grid = {
        'bb_period': [15, 20, 30],
        'bb_std': [1.5, 2.0, 2.5],
        'squeeze_period': [10, 15],
        'squeeze_percentile': [0.15, 0.20, 0.25],
        'volume_trigger': [1.2, 1.5, 2.0],
        'stop_mult': [1.0, 1.5, 2.0],
        'tp_mult': [3.0, 4.0, 5.0],
        'trail_trigger_mult': [0.5, 1.0, 1.5],
        'trail_mult': [1.0, 1.5, 2.0],
    }

    # 只用核心参数（减小搜索空间）
    core_params = [
        # (bb_period, bb_std, squeeze_pct, vol_trig, stop_m, tp_m, trail_trigger, trail_m)
        (20, 2.0, 0.20, 1.5, 1.5, 4.0, 1.0, 1.5),   # baseline
        (20, 2.0, 0.15, 1.5, 1.5, 4.0, 1.0, 1.5),   # tighter squeeze
        (20, 2.0, 0.25, 1.5, 1.5, 4.0, 1.0, 1.5),   # looser squeeze
        (20, 2.0, 0.20, 1.2, 1.5, 4.0, 1.0, 1.5),   # lower vol trigger
        (20, 2.0, 0.20, 2.0, 1.5, 4.0, 1.0, 1.5),   # higher vol trigger
        (20, 2.0, 0.20, 1.5, 1.0, 4.0, 1.0, 1.5),   # tighter stop
        (20, 2.0, 0.20, 1.5, 2.0, 4.0, 1.0, 1.5),   # wider stop
        (20, 2.0, 0.20, 1.5, 1.5, 3.0, 1.0, 1.5),   # lower TP
        (20, 2.0, 0.20, 1.5, 1.5, 5.0, 1.0, 1.5),   # higher TP
        (20, 2.0, 0.20, 1.5, 1.5, 4.0, 0.5, 1.5),   # earlier trail
        (20, 2.0, 0.20, 1.5, 1.5, 4.0, 1.5, 1.5),   # later trail
        (20, 2.0, 0.20, 1.5, 1.5, 4.0, 1.0, 1.0),   # tighter trail distance
        (20, 2.0, 0.20, 1.5, 1.5, 4.0, 1.0, 2.0),   # wider trail distance
        (15, 2.0, 0.20, 1.5, 1.5, 4.0, 1.0, 1.5),   # shorter BB
        (30, 2.0, 0.20, 1.5, 1.5, 4.0, 1.0, 1.5),   # longer BB
        (20, 1.5, 0.20, 1.5, 1.5, 4.0, 1.0, 1.5),   # narrower BB std
        (20, 2.5, 0.20, 1.5, 1.5, 4.0, 1.0, 1.5),   # wider BB std
        (20, 2.0, 0.20, 1.5, 1.5, 2.0, 1.0, 1.5),   # TP = 2x ATR (tight)
        (20, 2.0, 0.15, 1.2, 2.0, 4.0, 1.0, 1.5),   # combo: tight squeeze + low vol + wide stop
        (20, 2.0, 0.15, 2.0, 2.0, 4.0, 1.0, 1.5),   # combo: tight squeeze + high vol + wide stop
    ]

    results = []
    for i, (bb_p, bb_s, sq_pct, vol_t, stop_m, tp_m, trail_trig, trail_m) in enumerate(core_params):
        params = {
            'bb_period': bb_p,
            'bb_std': bb_s,
            'squeeze_percentile': sq_pct,
            'volume_trigger': vol_t,
            'stop_mult': stop_m,
            'tp_mult': tp_m,
            'trail_trigger_mult': trail_trig,
            'trail_mult': trail_m,
        }

        config = BacktestConfig(
            symbol='ETH/USDT',
            timeframe='1m',
            initial_capital=100.0,
            leverage=10,
            start_date='2026-06-24',
            end_date='2026-07-01',
            strategy_name='vol_breakout_1m',
            strategy_params=params,
        )

        try:
            strategy_cls = get_strategy(config.strategy_name)
            strategy = strategy_cls(config.strategy_params)
            engine = BacktestEngine(config)
            result = engine.run(strategy, df.copy())

            m = result.metrics
            results.append({
                'params': params,
                'return': m.total_return_pct,
                'sharpe': m.sharpe_ratio,
                'max_dd': m.max_drawdown_pct,
                'win_rate': m.win_rate,
                'trades': m.total_trades,
                'liquidations': m.liquidations,
                'profit_factor': m.profit_factor,
                'expectancy': m.expectancy,
                'score': (m.total_return_pct
                          - abs(m.max_drawdown_pct) * 0.3
                          + m.sharpe_ratio * 3
                          + m.win_rate * 100 * 0.05
                          + m.profit_factor * 2),
            })
        except Exception as e:
            print(f'  [跳过] {params}: {e}')

    # 排序
    results.sort(key=lambda r: r['score'], reverse=True)

    print(f'\n{"="*100}')
    print(f'TOP 10 参数组合 (按综合评分)')
    print(f'{"="*100}')
    print(f'{"#":<3} {"Return":>8} {"Sharpe":>7} {"MaxDD":>7} {"Win%":>7} '
          f'{"Trades":>6} {"Liq":>4} {"PF":>6} {"Exp":>7}  Params')
    print('-' * 100)
    for i, r in enumerate(results[:10]):
        p = r['params']
        p_str = (f'BB={p["bb_period"]}/{p["bb_std"]}σ sq={p["squeeze_percentile"]:.0%} '
                 f'vol>{p["volume_trigger"]}x '
                 f'SL={p["stop_mult"]}x TP={p["tp_mult"]}x '
                 f'Trail@{p["trail_trigger_mult"]}x/{p["trail_mult"]}x')
        print(f'{i+1:<3} {r["return"]:>+7.1f}% {r["sharpe"]:>+6.2f} {r["max_dd"]:>6.1f}% '
              f'{r["win_rate"]*100:>6.1f}% {r["trades"]:>5} {r["liquidations"]:>4} '
              f'{r["profit_factor"]:>5.2f} {r["expectancy"]:>+6.2f}%  {p_str}')

    print(f'\n共测试 {len(results)} 组参数，{sum(1 for r in results if r["return"] > 0)} 组正收益')

    return results


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    t0 = time.time()

    # ---- 网格搜索 ----
    all_results = grid_search()

    # ---- 最佳参数详细回测 ----
    if all_results:
        best = all_results[0]
        best_params = best['params']
        print(f'\n\n╔══════════════════════════════════════════════════╗')
        print(f'║ 🏆 最佳参数详细回测                               ║')
        print(f'╚══════════════════════════════════════════════════╝')

        run_single(
            config_overrides={
                'strategy_name': 'vol_breakout_1m',
                'symbol': 'ETH/USDT',
                'timeframe': '1m',
                'start_date': '2026-06-24',
                'end_date': '2026-07-01',
                'leverage': 10,
                'initial_capital': 100.0,
                'strategy_params': best_params,
            },
            strategy_params=best_params,
            label='最佳参数组合',
        )

    print(f'\n⏱️ 总耗时: {time.time()-t0:.1f}s')
