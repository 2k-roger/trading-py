#!/usr/bin/env python3
"""1m 策略 v2 回测 — 波动扩张剥头皮 + 参数网格搜索。

用法:
    python examples/backtest_1m_v2.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.config import ConfigLoader
from src.data.loader import DataLoader
from src.backtest.engine import BacktestEngine
from src.models import BacktestConfig

# 注册策略
import src.strategies.vol_scalp_1m  # noqa: F401
from src.strategies.registry import get_strategy

_REASON_CN = {
    'stop_loss': '初始止损', 'take_profit': '止盈',
    'trailing_stop': '移动止损', 'liquidation': '爆仓',
    'end_of_data': '数据结束', 'signal': '信号反转',
}


def run_config(loader, df, params, leverage, label=''):
    """单次回测。"""
    config = BacktestConfig(
        symbol='ETH/USDT', timeframe='1m',
        initial_capital=100.0, leverage=leverage,
        start_date='2026-06-24', end_date='2026-07-01',
        strategy_name='vol_scalp_1m', strategy_params=params,
    )
    try:
        strategy = get_strategy(config.strategy_name)(config.strategy_params)
        engine = BacktestEngine(config)
        result = engine.run(strategy, df.copy())
        m = result.metrics
        return {
            'params': params, 'leverage': leverage,
            'return': m.total_return_pct, 'sharpe': m.sharpe_ratio,
            'max_dd': m.max_drawdown_pct, 'win_rate': m.win_rate,
            'trades': m.total_trades, 'liquidations': m.liquidations,
            'profit_factor': m.profit_factor, 'expectancy': m.expectancy,
            'cagr': m.cagr,
            'score': (m.total_return_pct
                      - abs(m.max_drawdown_pct) * 0.5         # 惩罚大回撤
                      + m.sharpe_ratio * 2
                      + m.win_rate * 100 * 0.1
                      + m.profit_factor * 2
                      + min(m.total_trades, 30) * 0.2),       # 奖励交易量
            'result': result,
        }
    except Exception as e:
        return None


def print_trade_details(result):
    """打印逐笔交易。"""
    if not result.trades:
        print('  (无交易)')
        return
    print(f'  {"#":<3} {"入场":<20} {"出场":<20} '
          f'{"方向":<6} {"入场价":>10} {"出场价":>10} {"盈亏%":>8} {"原因":<12}')
    print(f'  {"-"*88}')
    for i, t in enumerate(result.trades, 1):
        d = '做多' if t.type == 'long' else '做空'
        r = _REASON_CN.get(t.exit_reason, t.exit_reason)
        print(f'  {i:<3} {str(t.entry_time):<20} {str(t.exit_time):<20} '
              f'{d:<6} {t.entry_price:>10.2f} {t.exit_price:>10.2f} '
              f'{t.pnl_pct:>+7.2f}% {r:<12}')


# ================================================================
print('╔══════════════════════════════════════════════════════════╗')
print('║ 1m 波动扩张剥头皮 v2 — 参数搜索                          ║')
print('╚══════════════════════════════════════════════════════════╝')

loader = DataLoader('binance')
df = loader.fetch(
    symbol='ETH/USDT', timeframe='1m',
    start_date='2026-06-24', end_date='2026-07-01',
)
print(f'数据: {len(df):,} 根 1m K线 ({df.index[0]} → {df.index[-1]})\n')

# ── 参数网格 ──
param_grid = [
    # (bb_period, bb_std, stop_m, tp_m, trail_m, vol_thr, range_pct)
    # --- Baseline variants ---
    (20, 2.0, 1.0, 2.0, 1.0, 1.5, 0.85),
    (20, 2.0, 1.0, 2.5, 1.0, 1.5, 0.85),
    (20, 2.0, 1.0, 1.5, 1.0, 1.5, 0.85),
    (20, 2.0, 1.5, 2.0, 1.0, 1.5, 0.85),
    (20, 2.0, 0.75, 2.0, 1.0, 1.5, 0.85),
    # --- BB variations ---
    (15, 2.0, 1.0, 2.0, 1.0, 1.5, 0.85),
    (30, 2.0, 1.0, 2.0, 1.0, 1.5, 0.85),
    (20, 1.5, 1.0, 2.0, 1.0, 1.5, 0.85),
    (20, 2.5, 1.0, 2.0, 1.0, 1.5, 0.85),
    # --- Volume threshold ---
    (20, 2.0, 1.0, 2.0, 1.0, 1.2, 0.85),
    (20, 2.0, 1.0, 2.0, 1.0, 2.0, 0.85),
    # --- Range percentile ---
    (20, 2.0, 1.0, 2.0, 1.0, 1.5, 0.80),
    (20, 2.0, 1.0, 2.0, 1.0, 1.5, 0.90),
    # --- Trail variations ---
    (20, 2.0, 1.0, 2.0, 0.75, 1.5, 0.85),
    (20, 2.0, 1.0, 2.0, 1.5, 1.5, 0.85),
    # --- Combo: tight stop + quick TP ---
    (15, 2.0, 0.75, 1.5, 0.75, 2.0, 0.90),
    # --- Combo: wide stop + big TP (fat tail capture) ---
    (20, 2.0, 1.5, 3.0, 1.5, 1.5, 0.85),
    # --- Combo: aggressive scalping ---
    (20, 2.0, 0.75, 1.5, 0.75, 1.2, 0.80),
    # --- Combo: conservative ---
    (30, 2.5, 1.5, 2.5, 1.0, 2.0, 0.90),
    # --- Combo: optimized for win rate ---
    (20, 2.0, 1.5, 1.5, 0.75, 1.5, 0.85),
]

# 杠杆范围
leverages = [3, 5, 10]

all_results = []
for lev in leverages:
    print(f'--- 杠杆 {lev}x ---')
    for p in param_grid:
        params = {
            'bb_period': p[0], 'bb_std': p[1],
            'stop_mult': p[2], 'tp_mult': p[3],
            'trail_mult': p[4], 'vol_threshold': p[5],
            'range_percentile': p[6],
        }
        r = run_config(loader, df, params, lev)
        if r:
            all_results.append(r)
    print(f'  累计 {len(all_results)} 组\n')

# ── 排序 ──
all_results.sort(key=lambda r: r['score'], reverse=True)

print('=' * 105)
print(f'TOP 15 参数组合 (综合评分)')
print('=' * 105)
hdr = f'{"#":<3} {"Lev":>3} {"Return":>8} {"Sharpe":>7} {"MaxDD":>7} {"Win%":>7} {"Trades":>6} {"Liq":>4} {"PF":>6} {"Exp":>7}  Params'
print(hdr)
print('-' * 105)
for i, r in enumerate(all_results[:15]):
    p = r['params']
    p_str = (f'BB={p["bb_period"]}/{p["bb_std"]}σ '
             f'SL={p["stop_mult"]}x TP={p["tp_mult"]}x '
             f'Trail={p["trail_mult"]}x '
             f'Vol>{p["vol_threshold"]}x Rng>{p["range_percentile"]:.0%}')
    print(f'{i+1:<3} {r["leverage"]:>3}x {r["return"]:>+7.1f}% {r["sharpe"]:>+6.2f} '
          f'{r["max_dd"]:>6.1f}% {r["win_rate"]*100:>6.1f}% {r["trades"]:>5} '
          f'{r["liquidations"]:>4} {r["profit_factor"]:>5.2f} {r["expectancy"]:>+6.2f}%  {p_str}')

print(f'\n共 {len(all_results)} 组，{sum(1 for r in all_results if r["return"] > 0)} 组正收益')

# ── 最佳组合详细 ──
if all_results:
    best = all_results[0]
    print(f'\n{"="*70}')
    print(f'🏆 最佳组合详细回测')
    print(f'{"="*70}')
    print(f'参数: {best["params"]}')
    print(f'杠杆: {best["leverage"]}x')
    print(f'收益率: {best["return"]:+.1f}% | 夏普: {best["sharpe"]:.2f} | '
          f'最大回撤: {best["max_dd"]:.1f}% | 胜率: {best["win_rate"]*100:.1f}%')
    print(f'交易: {best["trades"]}笔 | 爆仓: {best["liquidations"]} | '
          f'盈亏比: {best["profit_factor"]:.2f} | 期望: {best["expectancy"]:+.2f}%')
    print(f'\n逐笔交易:')
    print_trade_details(best['result'])

    # 分类统计
    trades = best['result'].trades
    longs = [t for t in trades if t.type == 'long']
    shorts = [t for t in trades if t.type == 'short']
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f'\n方向分布: 做多 {len(longs)} 笔, 做空 {len(shorts)} 笔')
    print(f'出场分布: {reasons}')
    if longs:
        print(f'做多胜率: {sum(1 for t in longs if t.pnl_pct>0)/len(longs)*100:.0f}%')
        print(f'做多平均盈亏: {np.mean([t.pnl_pct for t in longs]):+.2f}%')
    if shorts:
        print(f'做空胜率: {sum(1 for t in shorts if t.pnl_pct>0)/len(shorts)*100:.0f}%')
        print(f'做空平均盈亏: {np.mean([t.pnl_pct for t in shorts]):+.2f}%')

    # 按 exit_reason 统计
    print(f'\n出场原因分析:')
    for reason in ['take_profit', 'trailing_stop', 'stop_loss', 'liquidation']:
        subset = [t for t in trades if t.exit_reason == reason]
        if subset:
            avg_pnl = np.mean([t.pnl_pct for t in subset])
            print(f'  {_REASON_CN.get(reason, reason):<12}: {len(subset):>3} 笔, '
                  f'平均盈亏 {avg_pnl:+.2f}%')

print(f'\n⏱️ 总耗时: {time.time()-t0:.1f}s')
