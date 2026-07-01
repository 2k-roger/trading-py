#!/usr/bin/env python3
"""5m 策略 v2 — 扩展时间范围 + 聚焦 breakout 优化。

用法:
    python examples/backtest_5m_v2.py
"""

import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
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

# ═══════════════════════════════════════════════════════════
# 多时间段测试
# ═══════════════════════════════════════════════════════════
PERIODS = [
    ('近1周 (下跌)', '2026-06-24', '2026-07-01'),
    ('近2周',       '2026-06-17', '2026-07-01'),
    ('近1月',       '2026-06-01', '2026-07-01'),
]

def run_config(loader, params, leverage, start, end):
    """单次回测。"""
    df = loader.fetch(symbol='ETH/USDT', timeframe='5m',
                      start_date=start, end_date=end)
    config = BacktestConfig(
        symbol='ETH/USDT', timeframe='5m',
        initial_capital=100.0, leverage=leverage,
        start_date=start, end_date=end,
        strategy_name='momentum_scalp_5m',
        strategy_params=params,
        risk_per_trade_pct=0.005,  # 降低单笔风险到 0.5%
    )
    try:
        strategy = get_strategy(config.strategy_name)(config.strategy_params)
        engine = BacktestEngine(config)
        result = engine.run(strategy, df.copy())
        m = result.metrics
        return {
            'return': m.total_return_pct, 'sharpe': m.sharpe_ratio,
            'max_dd': m.max_drawdown_pct, 'win_rate': m.win_rate,
            'trades': m.total_trades, 'liquidations': m.liquidations,
            'profit_factor': m.profit_factor, 'expectancy': m.expectancy,
            'cagr': m.cagr, 'n_bars': len(df),
            'result': result,
        }
    except:
        return None

def print_summary(r, label=''):
    if r is None:
        print(f'  {label}: 回测失败')
        return
    print(f'  {label}: {r["n_bars"]:>5}根 | '
          f'收益={r["return"]:>+6.1f}% | 夏普={r["sharpe"]:>+5.1f} | '
          f'回撤={r["max_dd"]:>5.1f}% | 胜率={r["win_rate"]*100:>4.0f}% | '
          f'交易={r["trades"]:>3}笔 | PF={r["profit_factor"]:.2f} | '
          f'爆仓={r["liquidations"]}')

# ═══════════════════════════════════════════════════════════
print('╔══════════════════════════════════════════════════════════╗')
print('║  5m 策略 v2 — 多时间段验证 + breakout 优化               ║')
print('╚══════════════════════════════════════════════════════════╝')

loader = DataLoader('binance')

# ── 候选参数（聚焦 breakout 和 momentum） ──
candidates = [
    # (mode, ema_f, ema_s, sl, tp, tr_trig, tr_m, vol, lb, leverage)
    # v1 最佳 breakout
    ('breakout', 20, 50, 1.5, 2.0, 0.75, 1.0, 1.2, 5, 5),
    ('breakout', 20, 50, 2.0, 3.0, 1.0, 1.5, 1.5, 5, 5),
    # 优化版 breakout (更紧止损, 更大止盈)
    ('breakout', 20, 50, 1.5, 3.0, 0.75, 1.0, 1.5, 5, 5),
    ('breakout', 20, 50, 2.0, 4.0, 1.0, 1.5, 1.5, 5, 3),
    # momentum (少而精)
    ('momentum', 20, 50, 2.0, 2.5, 1.0, 1.5, 1.5, 5, 5),
    ('momentum', 20, 50, 1.5, 2.0, 0.75, 1.0, 1.5, 5, 3),
    # 高胜率导向
    ('momentum', 20, 100, 2.5, 3.0, 1.5, 1.5, 2.0, 5, 3),
    # all modes
    ('all', 20, 50, 2.0, 3.0, 1.0, 1.5, 1.5, 5, 3),
]

print(f'\n测试 {len(PERIODS)} 个时间段 × {len(candidates)} 组参数')
print('=' * 90)

best_overall = None
best_score = -999

for period_name, start, end in PERIODS:
    print(f'\n── {period_name} ({start} → {end}) ──')
    period_best = None
    period_best_score = -999

    for c in candidates:
        mode, ef, es, sl, tp, tr_trig, tr_m, vol, lb, lev = c
        params = {
            'ema_fast': ef, 'ema_slow': es,
            'stop_mult': sl, 'tp_mult': tp,
            'trail_trigger': tr_trig, 'trail_mult': tr_m,
            'vol_threshold': vol, 'lookback_break': lb,
            'entry_mode': mode,
        }
        r = run_config(loader, params, lev, start, end)
        if r:
            # 评分
            score = (r['return']
                     - abs(r['max_dd']) * 0.3
                     + r['sharpe'] * 1.5
                     + r['profit_factor'] * 1.0
                     + (1.0 if r['return'] > 0 else 0))
            if score > period_best_score:
                period_best_score = score
                period_best = (c, r)
            if score > best_score:
                best_score = score
                best_overall = (period_name, c, r)

            marker = '✅' if r['return'] > 0 else '  '
            print(f'  {marker} {mode:<10} EMA({ef},{es}) SL={sl}× TP={tp}× '
                  f'Tr={tr_trig}/{tr_m}× Vol>{vol}× LB={lb} Lev={lev}× | '
                  f'R={r["return"]:>+5.1f}% S={r["sharpe"]:>+4.1f} '
                  f'DD={r["max_dd"]:>5.1f}% WR={r["win_rate"]*100:>4.0f}% '
                  f'T={r["trades"]:>3} PF={r["profit_factor"]:.1f}')

    # 每时间段最佳
    if period_best:
        c, r = period_best
        print(f'  🏆 最佳: {c[0]} R={r["return"]:+.1f}% T={r["trades"]}笔')

# ═══════════════════════════════════════════════════════════
# 最佳组合详细回测
# ═══════════════════════════════════════════════════════════
if best_overall:
    period_name, c, _ = best_overall
    mode, ef, es, sl, tp, tr_trig, tr_m, vol, lb, lev = c
    params = {
        'ema_fast': ef, 'ema_slow': es,
        'stop_mult': sl, 'tp_mult': tp,
        'trail_trigger': tr_trig, 'trail_mult': tr_m,
        'vol_threshold': vol, 'lookback_break': lb,
        'entry_mode': mode,
    }

    print(f'\n\n{"="*70}')
    print(f'🏆 全局最佳 — 在所有时间段回测')
    print(f'{"="*70}')

    for period_name, start, end in PERIODS:
        r = run_config(loader, params, lev, start, end)
        print_summary(r, period_name)

    print(f'\n最佳参数: {params}')
    print(f'杠杆: {lev}x | 风险/笔: 0.5%')

    # 最近一周的逐笔交易
    print(f'\n── 近1周 逐笔交易明细 ──')
    r = run_config(loader, params, lev, '2026-06-24', '2026-07-01')
    if r and r['result'].trades:
        trades = r['result'].trades
        print(f'{"#":<3} {"入场":<20} {"出场":<20} '
              f'{"方向":<6} {"入场价":>10} {"出场价":>10} {"盈亏%":>8} {"原因":<12} {"持仓":>5}')
        print('-' * 95)
        for i, t in enumerate(trades, 1):
            d = '做多' if t.type == 'long' else '做空'
            rc = _REASON_CN.get(t.exit_reason, t.exit_reason)
            print(f'{i:<3} {str(t.entry_time):<20} {str(t.exit_time):<20} '
                  f'{d:<6} {t.entry_price:>10.2f} {t.exit_price:>10.2f} '
                  f'{t.pnl_pct:>+7.2f}% {rc:<12} {t.holding_period:>4}根')

        longs = [t for t in trades if t.type == 'long']
        shorts = [t for t in trades if t.type == 'short']
        print(f'\n做多: {len(longs)}笔 胜率={sum(1 for t in longs if t.pnl_pct>0)/max(len(longs),1)*100:.0f}%')
        print(f'做空: {len(shorts)}笔 胜率={sum(1 for t in shorts if t.pnl_pct>0)/max(len(shorts),1)*100:.0f}%')
        for reason in ['take_profit', 'trailing_stop', 'stop_loss', 'liquidation']:
            subset = [t for t in trades if t.exit_reason == reason]
            if subset:
                print(f'{_REASON_CN.get(reason, reason)}: {len(subset)}笔 均盈亏={np.mean([t.pnl_pct for t in subset]):+.2f}%')

print(f'\n⏱️ 总耗时: {time.time()-t_start:.0f}s' if 't_start' in dir() else '')
