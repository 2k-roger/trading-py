"""5m 动量剥头皮策略 — 基于多周期分析的最优匹配。

设计依据（近7天 5m 分析）：
  - ACF(1)=+0.026, VR(10)=1.076 → 短期动量存在弱持续性
  - ATR=0.246%, 手续费=0.08% → 费用占 ATR×1.5 仅 22%，有利润空间
  - 峰度 13.8 → 肥尾，大波动后有余波
  - 平均连续同向 1.9 根, 90分位 4 根 → 持仓 1-3 根最优
  - ATR×1.5 止损触碰率 12.7% → 合理的止损距离
  - ATR×2.0 止损触碰率 5.3% → 安全距离

策略逻辑：
  1. 趋势过滤：EMA50 定大方向（仅顺势入场）
  2. 动量确认：价格突破近期高低点 + 成交量放大
  3. 入场执行：回调到 EMA20 附近后恢复 → 入场
  4. 快速止盈：ATR×2.0（捕捉 2-3 根K线的动量延续）
  5. 移动止损：盈利 0.75×ATR 后启动，保护利润
  6. 双向交易：做多/做空对称

三种入场模式（可配置）：
  - momentum: 突破近期高低点 + 放量
  - pullback: 趋势中回调到 EMA 后恢复
  - breakout: BB 收缩后扩张突破
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..models import TradeSetup, Position


@register('momentum_scalp_5m')
class MomentumScalp5M(Strategy):
    """5m 动量剥头皮。

    Parameters
    ----------
    ema_fast : int (20)       — 快线/回调基准
    ema_slow : int (50)       — 慢线/趋势过滤
    atr_period : int (14)
    stop_mult : float (1.5)   — 初始止损 ATR 倍数
    tp_mult : float (2.0)     — 止盈 ATR 倍数
    trail_trigger : float (0.75) — 移动止损触发距离（ATR倍数）
    trail_mult : float (1.0)  — 移动止损距离（ATR倍数）
    vol_threshold : float (1.2)  — 成交量放大阈值
    lookback_break : int (5)  — 突破回看K线数
    entry_mode : str ('momentum') — 'momentum' | 'pullback' | 'all'
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.ema_fast = self.params.get('ema_fast', 20)
        self.ema_slow = self.params.get('ema_slow', 50)
        self.atr_period = self.params.get('atr_period', 14)
        self.stop_mult = self.params.get('stop_mult', 1.5)
        self.tp_mult = self.params.get('tp_mult', 2.0)
        self.trail_trigger = self.params.get('trail_trigger', 0.75)
        self.trail_mult = self.params.get('trail_mult', 1.0)
        self.vol_threshold = self.params.get('vol_threshold', 1.2)
        self.lookback_break = self.params.get('lookback_break', 5)
        self.entry_mode = self.params.get('entry_mode', 'momentum')

    # ------------------------------------------------------------------
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ---- EMAs ----
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()

        # 趋势方向：价格在 EMA50 之上 = 多头趋势
        df['trend_up'] = (df['close'] > df['ema_slow']).astype(int)

        # 价格相对 EMA20 的距离
        df['dist_fast'] = (df['close'] - df['ema_fast']) / df['ema_fast'] * 100

        # ---- ATR ----
        prev = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev).abs(),
            (df['low'] - prev).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()

        # ---- 成交量 ----
        df['vol_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['vol_ratio'] = (df['volume'] / df['vol_ma']).fillna(1.0)

        # ---- 近期高低点 ----
        df['high_n'] = df['high'].rolling(window=self.lookback_break, min_periods=1).max()
        df['low_n'] = df['low'].rolling(window=self.lookback_break, min_periods=1).min()

        # 突破信号
        df['break_high'] = ((df['close'] > df['high_n'].shift(1))
                            .fillna(0).astype(int))
        df['break_low'] = ((df['close'] < df['low_n'].shift(1))
                           .fillna(0).astype(int))

        # ---- 回调检测 ----
        # 多头趋势中，价格从上方回到 EMA20 附近（±0.5×ATR 范围内）
        atr_approx = df['atr'].fillna(0)
        df['near_ema'] = (
            (df['close'] - df['ema_fast']).abs() <= atr_approx * 0.5
        ).astype(int)

        # 之前是否在 EMA20 之上（多头趋势下回调）
        df['was_above'] = (df['close'].shift(1) > df['ema_fast'].shift(1)).fillna(0).astype(int)
        df['was_below'] = (df['close'].shift(1) < df['ema_fast'].shift(1)).fillna(0).astype(int)

        # 回调信号：多头趋势 + 之前在上方 + 现在回到附近
        df['pullback_long'] = (
            (df['trend_up'] == 1) & (df['was_above'] == 1) & (df['near_ema'] == 1)
        ).astype(int)

        # 空头趋势 + 之前在下方 + 现在回到附近
        df['pullback_short'] = (
            (df['trend_up'] == 0) & (df['was_below'] == 1) & (df['near_ema'] == 1)
        ).astype(int)

        # ---- BB 收缩（辅助） ----
        bb_mid = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_width'] = (bb_std * 4) / bb_mid * 100  # 带宽

        # 收缩信号：带宽在 30 周期最低 20%
        df['bb_squeeze'] = (
            df['bb_width']
            .rolling(window=30, min_periods=20)
            .rank(pct=True) <= 0.20
        ).fillna(0).astype(int)

        # 扩张突破
        df['bb_break_up'] = (
            (df['bb_squeeze'] == 1) &
            (df['close'] > bb_mid + bb_std * 2) &
            (df['close'].shift(1) <= (bb_mid.shift(1) + bb_std.shift(1) * 2))
        ).fillna(0).astype(int)

        df['bb_break_dn'] = (
            (df['bb_squeeze'] == 1) &
            (df['close'] < bb_mid - bb_std * 2) &
            (df['close'].shift(1) >= (bb_mid.shift(1) - bb_std.shift(1) * 2))
        ).fillna(0).astype(int)

        return df

    # ------------------------------------------------------------------
    def on_bar(self, df: pd.DataFrame, idx: int, position: Optional[Position]) -> TradeSetup:
        if position is not None:
            return TradeSetup(action='none')

        row = df.iloc[idx]
        atr = row.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return TradeSetup(action='none')
        if idx < max(self.ema_slow, 30):
            return TradeSetup(action='none')

        close = row['close']
        vol_ratio = row.get('vol_ratio', 1.0)
        entry_mode = self.entry_mode

        # ═══════════════════════════════════════════════════════
        # 模式 1: 动量突破（核心）
        # ═══════════════════════════════════════════════════════
        if entry_mode in ('momentum', 'all'):
            # 做多：突破 N 周期高点 + 放量 + 多头趋势
            if (row.get('break_high', 0) == 1
                    and vol_ratio >= self.vol_threshold
                    and row.get('trend_up', 0) == 1):
                stop = close - self.stop_mult * atr
                tp = close + self.tp_mult * atr
                return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

            # 做空：跌破 N 周期低点 + 放量 + 空头趋势
            if (row.get('break_low', 0) == 1
                    and vol_ratio >= self.vol_threshold
                    and row.get('trend_up', 0) == 0):
                stop = close + self.stop_mult * atr
                tp = close - self.tp_mult * atr
                return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        # ═══════════════════════════════════════════════════════
        # 模式 2: 回调入场（保守）
        # ═══════════════════════════════════════════════════════
        if entry_mode in ('pullback', 'all'):
            # 做多回调：多头趋势 + 回到 EMA20 + 价格开始反弹（当前收盘 > 开盘）
            if (row.get('pullback_long', 0) == 1
                    and close > df.iloc[idx]['open']
                    and vol_ratio >= 1.0):
                stop = close - self.stop_mult * atr
                tp = close + self.tp_mult * atr
                return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

            # 做空回调
            if (row.get('pullback_short', 0) == 1
                    and close < df.iloc[idx]['open']
                    and vol_ratio >= 1.0):
                stop = close + self.stop_mult * atr
                tp = close - self.tp_mult * atr
                return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        # ═══════════════════════════════════════════════════════
        # 模式 3: BB 收缩突破（低胜率但高盈亏比）
        # ═══════════════════════════════════════════════════════
        if entry_mode in ('breakout', 'all'):
            if row.get('bb_break_up', 0) == 1 and vol_ratio >= self.vol_threshold:
                stop = close - self.stop_mult * 1.5 * atr  # 宽止损
                tp = close + self.tp_mult * 2.0 * atr      # 大止盈
                return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

            if row.get('bb_break_dn', 0) == 1 and vol_ratio >= self.vol_threshold:
                stop = close + self.stop_mult * 1.5 * atr
                tp = close - self.tp_mult * 2.0 * atr
                return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        return TradeSetup(action='none')

    # ------------------------------------------------------------------
    def get_trailing_stop(self, position: Position, bar) -> float:
        """阶梯式移动止损 — 盈利触发后快速收紧。"""
        atr = bar.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return position.stop_loss

        current_stop = position.stop_loss
        trigger_dist = self.trail_trigger * atr

        if position.type == 'long':
            profit_dist = bar['close'] - position.entry_price
            if profit_dist >= trigger_dist:
                # 第一阶梯：移到 盈亏平衡 + 少许利润
                breakeven_stop = position.entry_price + 0.1 * atr
                trail_stop = bar['close'] - self.trail_mult * atr
                # 如果盈利超过 tp_mult*0.5，进一步收紧
                if profit_dist >= self.tp_mult * atr * 0.5:
                    trail_stop = bar['close'] - self.trail_mult * 0.5 * atr
                return max(current_stop, trail_stop, breakeven_stop)
            return current_stop
        else:
            profit_dist = position.entry_price - bar['close']
            if profit_dist >= trigger_dist:
                breakeven_stop = position.entry_price - 0.1 * atr
                trail_stop = bar['close'] + self.trail_mult * atr
                if profit_dist >= self.tp_mult * atr * 0.5:
                    trail_stop = bar['close'] + self.trail_mult * 0.5 * atr
                return min(current_stop, trail_stop, breakeven_stop)
            return current_stop
