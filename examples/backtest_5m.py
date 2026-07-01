#!/usr/bin/env python3
"""5m 动量剥头皮策略 — 完整回测 & 参数网格搜索。

用法:
    python examples/backtest_5m.py
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

import src.strategies.momentum_scalp_5m  # noqa: F401
from src.strategies.registry import get_strategy

_REASON_CN = {
    'stop_loss': '初始止损', 'take_profit': '止盈',
    'trailing_stop': '移动止损', 'liquidation': '爆仓',
    'end_of_data': '数据结束', 'signal': '信号反转',
}


def run_one(loader, df, params, leverage, label=''):
    """单次回测，返回指标字典 + result 对象。"""
    config = BacktestConfig(
        symbol='ETH/USDT', timeframe='5m',
        initial_capital=100.0, leverage=leverage,
        start_date='2026-06-24', end_date='2026-07-01',
        strategy_name='momentum_scalp_5m', strategy_params=params,
    )
    try:
        strategy = get_strategy(config.strategy_name)(config.strategy_params)
        engine = BacktestEngine(config)
        result = engine.run(strategy, df.copy())
        m = result.metrics
        # 综合评分：奖励收益+交易量+盈亏比，惩罚回撤
        score = (m.total_return_pct
                 - abs(m.max_drawdown_pct) * 0.4
                 + m.sharpe_ratio * 1.5
                 + m.win_rate * 100 * 0.08
                 + m.profit_factor * 1.5
                 + min(m.total_trades, 40) * 0.15
                 + (1.0 if m.total_return_pct > 0 else 0.0))
        return {
            'params': params, 'leverage': leverage,
            'return': m.total_return_pct, 'sharpe': m.sharpe_ratio,
            'max_dd': m.max_drawdown_pct, 'win_rate': m.win_rate,
            'trades': m.total_trades, 'liquidations': m.liquidations,
            'profit_factor': m.profit_factor, 'expectancy': m.expectancy,
            'cagr': m.cagr, 'score': score, 'result': result,
        }
    except Exception as e:
        return None


# ================================================================
print('╔══════════════════════════════════════════════════════════╗')
print('║  5m 动量剥头皮 — 参数搜索 & 对比                         ║')
print('╚══════════════════════════════════════════════════════════╝')

loader = DataLoader('binance')
df = loader.fetch(
    symbol='ETH/USDT', timeframe='5m',
    start_date='2026-06-24', end_date='2026-07-01',
)
print(f'\n数据: {len(df):,} 根 5m K线 ({df.index[0]} → {df.index[-1]})\n')

t_start = time.time()

# ── 参数网格 ──
# 结构: (ema_fast, ema_slow, stop_m, tp_m, trail_trig, trail_m, vol_thr, lookback, mode)
param_sets = [
    # --- Momentum mode (core) ---
    (20, 50, 1.5, 2.0, 0.75, 1.0, 1.2, 5, 'momentum'),
    (20, 50, 1.5, 2.5, 0.75, 1.0, 1.2, 5, 'momentum'),
    (20, 50, 1.5, 1.5, 0.75, 1.0, 1.2, 5, 'momentum'),
    (20, 50, 1.0, 2.0, 0.75, 1.0, 1.2, 5, 'momentum'),
    (20, 50, 2.0, 2.0, 0.75, 1.0, 1.2, 5, 'momentum'),
    # Lookback variations
    (20, 50, 1.5, 2.0, 0.75, 1.0, 1.2, 3, 'momentum'),
    (20, 50, 1.5, 2.0, 0.75, 1.0, 1.2, 8, 'momentum'),
    # EMA variations
    (12, 50, 1.5, 2.0, 0.75, 1.0, 1.2, 5, 'momentum'),
    (30, 50, 1.5, 2.0, 0.75, 1.0, 1.2, 5, 'momentum'),
    (20, 30, 1.5, 2.0, 0.75, 1.0, 1.2, 5, 'momentum'),
    (20, 100, 1.5, 2.0, 0.75, 1.0, 1.2, 5, 'momentum'),
    # Volume threshold
    (20, 50, 1.5, 2.0, 0.75, 1.0, 1.0, 5, 'momentum'),
    (20, 50, 1.5, 2.0, 0.75, 1.0, 1.5, 5, 'momentum'),
    # Trail variations
    (20, 50, 1.5, 2.0, 0.5, 1.0, 1.2, 5, 'momentum'),
    (20, 50, 1.5, 2.0, 1.0, 1.0, 1.2, 5, 'momentum'),
    (20, 50, 1.5, 2.0, 0.75, 0.75, 1.2, 5, 'momentum'),
    (20, 50, 1.5, 2.0, 0.75, 1.5, 1.2, 5, 'momentum'),

    # --- Pullback mode ---
    (20, 50, 1.5, 2.0, 0.5, 1.0, 1.0, 5, 'pullback'),
    (20, 50, 1.5, 2.5, 0.5, 1.0, 1.0, 5, 'pullback'),
    (20, 50, 2.0, 2.0, 0.75, 1.0, 1.0, 5, 'pullback'),
    (20, 50, 1.5, 2.0, 0.5, 1.0, 1.0, 5, 'pullback'),
    (20, 50, 1.0, 1.5, 0.5, 0.75, 1.0, 5, 'pullback'),

    # --- Breakout mode ---
    (20, 50, 1.5, 2.0, 0.75, 1.0, 1.2, 5, 'breakout'),
    (20, 50, 2.0, 3.0, 1.0, 1.5, 1.5, 5, 'breakout'),

    # --- All modes combined ---
    (20, 50, 1.5, 2.0, 0.75, 1.0, 1.2, 5, 'all'),
    (20, 50, 2.0, 2.5, 1.0, 1.0, 1.5, 5, 'all'),

    # --- Aggressive scalping ---
    (12, 50, 1.0, 1.5, 0.5, 0.75, 1.2, 3, 'momentum'),
    (20, 50, 0.75, 1.5, 0.5, 0.75, 1.0, 3, 'momentum'),

    # --- Conservative ---
    (20, 100, 2.0, 3.0, 1.0, 1.5, 1.5, 5, 'momentum'),
    (30, 100, 2.5, 3.0, 1.5, 1.5, 2.0, 8, 'all'),
]

leverages = [3, 5, 10]

all_results = []
for lev in leverages:
    for p_tuple in param_sets:
        params = {
            'ema_fast': p_tuple[0], 'ema_slow': p_tuple[1],
            'stop_mult': p_tuple[2], 'tp_mult': p_tuple[3],
            'trail_trigger': p_tuple[4], 'trail_mult': p_tuple[5],
            'vol_threshold': p_tuple[6], 'lookback_break': p_tuple[7],
            'entry_mode': p_tuple[8],
        }
        r = run_one(loader, df, params, lev)
        if r:
            all_results.append(r)

all_results.sort(key=lambda r: r['score'], reverse=True)

# ── 输出 ──
print(f'\n{"="*115}')
print(f'TOP 20 参数组合 (综合评分) — 共 {len(all_results)} 组，'
      f'{sum(1 for r in all_results if r["return"] > 0)} 组正收益')
print(f'{"="*115}')
hdr = (f'{"#":<3} {"Lev":>3} {"Return":>8} {"Sharpe":>7} {"MaxDD":>7} '
       f'{"Win%":>7} {"Trades":>6} {"Liq":>4} {"PF":>6} {"Exp":>7}  Mode       Params')
print(hdr)
print('-' * 115)

for i, r in enumerate(all_results[:20]):
    p = r['params']
    p_str = (f'EMA({p["ema_fast"]},{p["ema_slow"]}) '
             f'SL={p["stop_mult"]}x TP={p["tp_mult"]}x '
             f'Tr={p["trail_trigger"]}/{p["trail_mult"]}x '
             f'Vol>{p["vol_threshold"]}x LB={p["lookback_break"]}')
    print(f'{i+1:<3} {r["leverage"]:>3}x {r["return"]:>+7.1f}% {r["sharpe"]:>+6.2f} '
          f'{r["max_dd"]:>6.1f}% {r["win_rate"]*100:>6.1f}% {r["trades"]:>5} '
          f'{r["liquidations"]:>4} {r["profit_factor"]:>5.2f} {r["expectancy"]:>+6.2f}%  '
          f'{p["entry_mode"]:<10} {p_str}')

# ── 按模式分类统计 ──
print(f'\n{"="*60}')
print(f'按入场模式统计 (杠杆 5x，过滤 trades≥5)')
print(f'{"="*60}')
for mode in ['momentum', 'pullback', 'breakout', 'all']:
    subset = [r for r in all_results
              if r['params']['entry_mode'] == mode
              and r['leverage'] == 5
              and r['trades'] >= 5]
    if subset:
        avg_ret = np.mean([r['return'] for r in subset])
        avg_wr = np.mean([r['win_rate'] for r in subset]) * 100
        avg_pf = np.mean([r['profit_factor'] for r in subset])
        best = max(subset, key=lambda r: r['return'])
        print(f'  {mode:<12}: 平均收益={avg_ret:+.1f}% | 平均胜率={avg_wr:.0f}% | '
              f'平均PF={avg_pf:.2f} | '
              f'最佳={best["return"]:+.1f}% ({best["trades"]}笔, '
              f'Sharpe={best["sharpe"]:.1f})')
    else:
        print(f'  {mode:<12}: (样本不足)')

# ── 最佳组合详细 ──
if all_results:
    best = all_results[0]
    print(f'\n\n{"="*70}')
    print(f'🏆 最佳组合详细回测')
    print(f'{"="*70}')
    p = best['params']
    print(f'模式: {p["entry_mode"]} | 杠杆: {best["leverage"]}x')
    print(f'参数: EMA({p["ema_fast"]},{p["ema_slow"]}) '
          f'SL={p["stop_mult"]}×ATR TP={p["tp_mult"]}×ATR '
          f'Trail@{p["trail_trigger"]}/{p["trail_mult"]}×ATR '
          f'Vol>{p["vol_threshold"]}× LB={p["lookback_break"]}')
    print(f'收益率: {best["return"]:+.1f}% | 年化: {best["cagr"]:+.1f}% | '
          f'夏普: {best["sharpe"]:.2f} | 最大回撤: {best["max_dd"]:.1f}%')
    print(f'胜率: {best["win_rate"]*100:.1f}% | 交易: {best["trades"]}笔 | '
          f'爆仓: {best["liquidations"]} | 盈亏比: {best["profit_factor"]:.2f}')
    print(f'期望值: {best["expectancy"]:+.2f}%/笔')

    # 逐笔交易
    result = best['result']
    if result.trades:
        print(f'\n逐笔交易明细:')
        print(f'{"#":<3} {"入场":<20} {"出场":<20} '
              f'{"方向":<6} {"入场价":>10} {"出场价":>10} {"盈亏%":>8} {"原因":<12} {"持仓":>5}')
        print('-' * 95)
        for i, t in enumerate(result.trades, 1):
            d = '做多' if t.type == 'long' else '做空'
            r = _REASON_CN.get(t.exit_reason, t.exit_reason)
            print(f'{i:<3} {str(t.entry_time):<20} {str(t.exit_time):<20} '
                  f'{d:<6} {t.entry_price:>10.2f} {t.exit_price:>10.2f} '
                  f'{t.pnl_pct:>+7.2f}% {r:<12} {t.holding_period:>4}根')

        # 统计
        trades = result.trades
        longs = [t for t in trades if t.type == 'long']
        shorts = [t for t in trades if t.type == 'short']
        print(f'\n分析:')
        print(f'  做多: {len(longs)}笔 '
              f'(胜率 {sum(1 for t in longs if t.pnl_pct>0)/max(len(longs),1)*100:.0f}%, '
              f'均盈亏 {np.mean([t.pnl_pct for t in longs]) if longs else 0:+.2f}%)')
        print(f'  做空: {len(shorts)}笔 '
              f'(胜率 {sum(1 for t in shorts if t.pnl_pct>0)/max(len(shorts),1)*100:.0f}%, '
              f'均盈亏 {np.mean([t.pnl_pct for t in shorts]) if shorts else 0:+.2f}%)')

        reasons = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        for reason in ['take_profit', 'trailing_stop', 'stop_loss', 'liquidation']:
            cnt = reasons.get(reason, 0)
            subset = [t for t in trades if t.exit_reason == reason]
            avg_pnl = np.mean([t.pnl_pct for t in subset]) if subset else 0
            print(f'  {_REASON_CN.get(reason, reason):<12}: {cnt:>3}笔, 平均盈亏 {avg_pnl:+.2f}%')

        print(f'  平均持仓: {np.mean([t.holding_period for t in trades]):.1f} 根K线')

print(f'\n⏱️ 总耗时: {time.time()-t_start:.0f}s')
