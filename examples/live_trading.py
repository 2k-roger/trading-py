#!/usr/bin/env python3
"""ETH/USDT 实时模拟交易（WebSocket 推送）。

通过 ccxt.pro 订阅 Binance WebSocket K 线流，
Binance 主动推送每根 K 线，闭合时自动触发策略信号。

用法：
    python examples/live_trading.py
    python examples/live_trading.py --log-level DEBUG

退出：
    Ctrl+C 安全退出
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import ConfigLoader
from src.strategies.registry import get_strategy
from src.risk.manager import RiskManager
from src.models import Trade, Position

# 注册策略
import src.strategies.ema_crossover_v2  # noqa: F401

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
LOG_FILE = LOG_DIR / 'live_trading.log'
WARMUP_BARS = 300  # 指标预热所需历史 K 线数

_REASON_CN = {
    'stop_loss':     '初始止损',
    'take_profit':   '止盈',
    'trailing_stop': '移动止损',
    'liquidation':   '爆仓',
    'end_of_data':   '数据结束',
}


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('live')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    # 文件
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# 策略运行器
# ---------------------------------------------------------------------------
class LiveRunner:
    """逐 K 线运行策略，维护持仓和现金。"""

    def __init__(self, config, log: logging.Logger):
        self.config = config
        self.log = log
        self.risk = RiskManager(config)

        strategy_cls = get_strategy(config.strategy_name)
        self.strategy = strategy_cls(config.strategy_params)

        self.cash = config.initial_capital
        self.position: Position | None = None
        self.trade_count = 0
        self._df: pd.DataFrame | None = None  # 带指标的完整 DataFrame

    # ------------------------------------------------------------------
    def warmup(self, ohlcv_df: pd.DataFrame):
        """用历史数据初始化指标 DataFrame 和最后一条 K 线时间。"""
        self._df = self.strategy.compute_indicators(ohlcv_df)
        self.log.info('预热完成：%d 根 K 线就绪 (%s → %s)',
                      len(self._df), self._df.index[0], self._df.index[-1])

    # ------------------------------------------------------------------
    def on_closed_bar(self, bar: dict, ts: pd.Timestamp):
        """处理一根已闭合的 K 线。

        Args:
            bar: OHLCV 数据 {'open','high','low','close','volume': ...}
            ts: 时间戳
        """
        # ---- 追加到 DataFrame ----
        new_row = pd.DataFrame([bar], index=[ts])
        self._df = pd.concat([self._df, new_row])
        # 保持固定长度
        if len(self._df) > WARMUP_BARS * 2:
            self._df = self._df.iloc[-WARMUP_BARS:]

        # ---- 重新计算指标 ----
        self._df = self.strategy.compute_indicators(self._df)
        idx = len(self._df) - 1
        arrays = {col: self._df[col].values for col in self._df.columns}

        bar_high = float(arrays['high'][idx])
        bar_low = float(arrays['low'][idx])
        bar_close = float(arrays['close'][idx])

        self.log.info('─' * 50)
        self.log.info('新 K 线 %s | O=%.2f H=%.2f L=%.2f C=%.2f',
                      ts, bar['open'], bar_high, bar_low, bar_close)

        # ---- 1. 检查出场 ----
        if self.position is not None:
            exit_price, exit_reason = self._check_exits(bar_high, bar_low)
            if exit_price is not None:
                self._close(exit_price, exit_reason, ts, idx)

        # ---- 2. 检查入场 ----
        if self.position is None and self.cash > 0:
            setup = self.strategy.on_bar(self._df, idx, self.position)
            if setup.action in ('enter_long', 'enter_short'):
                pos = self._open(setup, arrays, idx, ts)
                if pos is not None:
                    entry_fee = self.risk.trade_fee(pos.position_value)
                    self.cash -= entry_fee + pos.margin
                    self.position = pos
                    self.log.info(
                        '%s 开仓 | %s | 价格=%.2f 数量=%.4f 保证金=%.4f '
                        '止损=%.2f 止盈=%s',
                        ts,
                        '做多' if pos.type == 'long' else '做空',
                        pos.entry_price, pos.size, pos.margin,
                        pos.stop_loss,
                        f'{pos.take_profit:.2f}' if not np.isnan(pos.take_profit) else '无',
                    )

        # ---- 3. 更新移动止损 ----
        if self.position is not None:
            from src.backtest.engine import FastBar
            bar_proxy = FastBar(arrays, idx)
            new_stop = self.strategy.get_trailing_stop(self.position, bar_proxy)
            if self.position.type == 'long' and new_stop > self.position.stop_loss:
                self.log.debug('移动止损上移：%.2f → %.2f', self.position.stop_loss, new_stop)
                self.position.stop_loss = new_stop
            elif self.position.type == 'short' and new_stop < self.position.stop_loss:
                self.log.debug('移动止损下移：%.2f → %.2f', self.position.stop_loss, new_stop)
                self.position.stop_loss = new_stop
            if bar_high > self.position.highest_price:
                self.position.highest_price = bar_high
            if bar_low < self.position.lowest_price:
                self.position.lowest_price = bar_low

        # ---- 4. 权益快照 ----
        equity = self.cash
        if self.position is not None:
            equity += self.position.unrealized_pnl(bar_close)
        self.log.debug('权益=$%.4f (现金=$%.4f %s)',
                       equity, self.cash,
                       '持仓中' if self.position else '空仓')

    # ------------------------------------------------------------------
    # 内部 — 复刻 BacktestEngine 逻辑
    # ------------------------------------------------------------------

    def _check_exits(self, bar_high, bar_low):
        pos = self.position
        slip = self.config.slippage_pct
        if pos.type == 'long':
            if bar_low <= pos.liquidation_price:
                return pos.liquidation_price, 'liquidation'
            if bar_low <= pos.stop_loss:
                price = pos.stop_loss * (1 - slip)
                reason = 'trailing_stop' if pos.stop_loss > pos.initial_stop_loss else 'stop_loss'
                return price, reason
            if not np.isnan(pos.take_profit) and bar_high >= pos.take_profit:
                return pos.take_profit * (1 - slip), 'take_profit'
        else:
            if bar_high >= pos.liquidation_price:
                return pos.liquidation_price, 'liquidation'
            if bar_high >= pos.stop_loss:
                price = pos.stop_loss * (1 + slip)
                reason = 'trailing_stop' if pos.stop_loss < pos.initial_stop_loss else 'stop_loss'
                return price, reason
            if not np.isnan(pos.take_profit) and bar_low <= pos.take_profit:
                return pos.take_profit * (1 + slip), 'take_profit'
        return None, None

    def _open(self, setup, arrays, idx, ts) -> Position | None:
        entry_price = float(arrays['close'][idx])
        size = self.risk.compute_position_size(
            self.cash, entry_price, setup.stop_loss,
            setup.action.replace('enter_', '')
        )
        if size <= 0:
            return None
        pos_value = entry_price * size
        margin = self.risk.required_margin(pos_value, self.config.leverage)
        entry_fee = self.risk.trade_fee(pos_value)
        if margin + entry_fee > self.cash:
            max_pv = self.cash / (1.0 / self.config.leverage + self.config.commission_pct)
            size = max_pv / entry_price
            if size <= 0:
                return None
            pos_value = entry_price * size
            margin = self.risk.required_margin(pos_value, self.config.leverage)
            entry_fee = self.risk.trade_fee(pos_value)
            if margin + entry_fee > self.cash * 1.001:
                return None

        side = 'long' if 'long' in setup.action else 'short'
        liq = self.risk.liquidation_price(entry_price, self.config.leverage, side)

        return Position(
            type=side, entry_time=ts, entry_price=entry_price, size=size,
            margin=margin, leverage=self.config.leverage, entry_idx=idx,
            stop_loss=setup.stop_loss or np.nan,
            initial_stop_loss=setup.stop_loss or np.nan,
            take_profit=setup.take_profit if setup.take_profit else np.nan,
            liquidation_price=liq,
            highest_price=float(arrays['high'][idx]),
            lowest_price=float(arrays['low'][idx]),
        )

    def _close(self, exit_price, reason, ts, idx):
        pos = self.position
        if reason == 'liquidation':
            pnl_abs = -pos.margin
        else:
            pnl_abs = pos.unrealized_pnl(exit_price)
            if pnl_abs < -pos.margin:
                pnl_abs = -pos.margin
        pnl_pct = (pnl_abs / pos.margin * 100) if pos.margin > 0 else 0.0
        exit_fee = self.risk.trade_fee(pos.position_value)
        released = pos.margin + pnl_abs - exit_fee
        self.cash += max(0.0, released)
        self.trade_count += 1

        reason_cn = _REASON_CN.get(reason, reason)
        direction = '做多' if pos.type == 'long' else '做空'
        self.log.info(
            '%s 平仓 | %s | 入场=%.2f 出场=%.2f | '
            '盈亏=%%%.2f ($%.4f) | %s | 余额=$%.4f | 累计%d笔',
            ts, direction,
            pos.entry_price, exit_price,
            pnl_pct, pnl_abs,
            reason_cn, self.cash, self.trade_count,
        )
        self.position = None

    # ------------------------------------------------------------------
    def status(self):
        equity = self.cash
        if self.position is not None:
            equity += self.position.unrealized_pnl(
                self._df['close'].iloc[-1] if self._df is not None else self.position.entry_price
            )
        pnl_pct = (equity / self.config.initial_capital - 1) * 100
        return (f'余额=$%.4f  权益=$%.4f (%+.2f%%)  '
                f'持仓=%s  累计=%d笔' % (
                    self.cash, equity, pnl_pct,
                    f'{self.position.type}@{self.position.entry_price:.2f}'
                    if self.position else '无',
                    self.trade_count))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description='ETH/USDT WebSocket 实时模拟交易')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING'])
    args = parser.parse_args()

    log = setup_logging(args.log_level)

    # ---- 配置 ----
    config = ConfigLoader.load(overrides={
        'strategy_name': 'ema_crossover_v2',
        'leverage': 15,
        'timeframe': '1h',
        'strategy_params': {
            'ema_short': 21, 'ema_long': 55, 'ema_trend': 100,
            'atr_period': 14, 'atr_mult': 0.25, 'tp_mult': 0.0,
        },
    })

    log.info('=' * 60)
    log.info('实时模拟交易启动 (WebSocket)')
    log.info('交易对：%s | 周期：%s | 杠杆：%dx | 初始资金：$%.2f',
             config.symbol, config.timeframe, config.leverage,
             config.initial_capital)
    log.info('策略：%s | 参数：%s', config.strategy_name, config.strategy_params)
    log.info('日志文件：%s', LOG_FILE)
    log.info('=' * 60)

    # ---- 历史数据预热 ----
    log.info('正在获取历史数据用于指标预热…')
    import ccxt  # noqa: E402
    rest = ccxt.binance({'enableRateLimit': True})
    raw_bars = rest.fetch_ohlcv(
        config.symbol, config.timeframe, limit=WARMUP_BARS
    )
    ohlcv_df = pd.DataFrame(
        raw_bars,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
    )
    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
    ohlcv_df.set_index('timestamp', inplace=True)
    ohlcv_df = ohlcv_df.astype(float)

    runner = LiveRunner(config, log)
    runner.warmup(ohlcv_df)

    # ---- WebSocket 订阅 ----
    exchange = ccxt.pro.binance()
    symbol_ws = config.symbol.replace('/', '')  # ETH/USDT → ETHUSDT

    log.info('连接 Binance WebSocket: %s@kline_%s …', symbol_ws, config.timeframe)
    log.info('等待 K 线推送 (Ctrl+C 退出)...\n')

    last_closed_ts = ohlcv_df.index[-1]

    try:
        while True:
            # watch_ohlcv 返回完整 OHLCV 列表，有新数据时才会 yield
            candles = await exchange.watch_ohlcv(symbol_ws, config.timeframe)

            if candles is None or len(candles) == 0:
                continue

            # 最新一根的 timestamp（ms → pd.Timestamp）
            latest = candles[-1]
            ts = pd.Timestamp(datetime.fromtimestamp(latest[0] / 1000, tz=timezone.utc)).tz_localize(None)

            # 只处理闭合的新 K 线
            if ts > last_closed_ts:
                bar = {
                    'open':   latest[1],
                    'high':   latest[2],
                    'low':    latest[3],
                    'close':  latest[4],
                    'volume': latest[5],
                }
                runner.on_closed_bar(bar, ts)
                last_closed_ts = ts

    except KeyboardInterrupt:
        log.info('\n收到退出信号…')
    except Exception as e:
        log.exception('运行异常：%s', e)
    finally:
        await exchange.close()
        log.info(runner.status())
        log.info('实时模拟交易已停止。日志文件：%s', LOG_FILE)


if __name__ == '__main__':
    import ccxt.pro  # noqa: E402 (ensure installed before asyncio.run)
    asyncio.run(main())
