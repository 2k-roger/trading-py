#!/usr/bin/env python3
"""ETH/USDT 量化回测脚本。

用法：
    python examples/run_backtest.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigLoader
from src.data.loader import DataLoader
from src.strategies.registry import get_strategy
from src.backtest.engine import BacktestEngine

# 导入策略模块以触发 @register 装饰器
import src.strategies.ema_crossover_atr  # noqa: F401
import src.strategies.ema_crossover_v2   # noqa: F401

# 出场原因中英文映射
_REASON_CN = {
    'stop_loss':     '初始止损',
    'take_profit':   '止盈',
    'trailing_stop': '移动止损',
    'liquidation':   '爆仓',
    'end_of_data':   '数据结束',
    'signal':        '信号反转',
}


def main():
    # ---- 1. 加载配置（默认值来自 config/default.yaml） ----
    # 针对 ETH/USDT 1h、2026-04 → 2026-06（下跌趋势）优化：
    #   EMA(21,55) 金叉死叉，ATR×0.25 移动止损，15x 杠杆
    #   不设止盈 — 让利润奔跑，靠极紧的移动止损锁利
    config = ConfigLoader.load(
        overrides={
            'strategy_name': 'ema_crossover_v2',
            'leverage': 15,
            'strategy_params': {
                'ema_short': 21,
                'ema_long': 55,
                'ema_trend': 100,
                'atr_period': 14,
                'atr_mult': 0.25,
                'tp_mult': 0.0,  # 不设止盈，纯移动止损出场
            },
        }
    )

    # ---- 2. 获取历史数据 ----
    loader = DataLoader(config.exchange)
    df = loader.fetch(
        symbol=config.symbol,
        timeframe=config.timeframe,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    print(f'已加载 {config.symbol} {config.timeframe} K线数据：{len(df):,} 根')
    print(f'时间范围：{df.index[0]} → {df.index[-1]}')

    # ---- 3. 实例化策略 ----
    strategy_cls = get_strategy(config.strategy_name)
    strategy = strategy_cls(config.strategy_params)
    print(f'交易策略：{config.strategy_name}')
    print(f'交易杠杆：{config.leverage}x')

    # ---- 4. 运行回测 ----
    engine = BacktestEngine(config)
    result = engine.run(strategy, df)

    # ---- 5. 打印绩效指标 ----
    print()
    print(result.metrics.display())

    final_equity = result.equity_curve['equity'].iloc[-1]
    pnl = final_equity - config.initial_capital
    print(f'初始资金：${config.initial_capital:.2f}')
    print(f'最终权益：${final_equity:.2f}')
    print(f'净盈亏：  ${pnl:+.2f} ({pnl/config.initial_capital*100:+.1f}%)')

    # ---- 6. 逐笔交易记录 ----
    if result.trades:
        print(f'\n{"序号":<5} {"入场时间":<20} {"出场时间":<20} '
              f'{"方向":<6} {"盈亏%":>8} {"出场原因":<12}')
        print('-' * 80)
        for i, t in enumerate(result.trades, 1):
            direction = '做多' if t.type == 'long' else '做空'
            reason_cn = _REASON_CN.get(t.exit_reason, t.exit_reason)
            print(
                f'{i:<5} {str(t.entry_time):<20} {str(t.exit_time):<20} '
                f'{direction:<6} {t.pnl_pct:>+7.2f}% {reason_cn:<12}'
            )


if __name__ == '__main__':
    main()
