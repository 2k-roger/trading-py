"""5m 价格突破策略 — 精简版。

基于多轮回测的结论：
  - BB收缩突破是唯一盈利的模式（+0.2%/周, 胜率58%, RR>2:1）
  - Momentum 模式交易过多（>100笔/周），被手续费吃掉
  - 简化设计，去掉无效模式，专注「通道突破 + 放量确认」

入场条件（OR 逻辑）：
  1. BB(20,2) 收缩后扩张突破 + 放量  → 高盈亏比
  2. Donchian(20) 通道突破 + 放量      → 补充信号
  3. 开盘区间突破（前 N 根最高/最低）  → 快速入场

出场：
  - 初始止损: ATR×1.5 (触碰率 ~12%)
  - 止盈: ATR×2.5 (盈亏比 1.67:1)
  - 阶梯移动止损: 盈利 0.5×ATR → 保本, 1.0×ATR → 收紧

风控：
  - 趋势过滤: EMA50 定方向（仅顺势入场）
  - 单笔风险: 0.5% 本金
  - 推荐杠杆: 3-5x
"""

from typing import Optional
import numpy as np
import pandas as pd
from .base import Strategy
from .registry import register
from ..models import TradeSetup, Position


@register('breakout_5m')
class Breakout5M(Strategy):
    """5m 多通道突破策略。

    Parameters
    ----------
    bb_period : int (20)
    bb_std : float (2.0)
    donchian_period : int (20)
    ema_trend : int (50)
    atr_period : int (14)
    stop_mult : float (1.5)
    tp_mult : float (2.5)
    trail_start : float (0.5)   — 启动移动止损的盈利阈值 (ATR倍数)
    trail_tight : float (1.0)   — 第一阶梯移动止损距离
    trail_tighter : float (0.5) — 第二阶梯（盈利 > tp_mult*0.5 后）
    vol_threshold : float (1.2)
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std = self.params.get('bb_std', 2.0)
        self.donchian = self.params.get('donchian_period', 20)
        self.ema_trend = self.params.get('ema_trend', 50)
        self.atr_period = self.params.get('atr_period', 14)
        self.stop_mult = self.params.get('stop_mult', 1.5)
        self.tp_mult = self.params.get('tp_mult', 2.5)
        self.trail_start = self.params.get('trail_start', 0.5)
        self.trail_tight = self.params.get('trail_tight', 1.0)
        self.trail_tighter = self.params.get('trail_tighter', 0.5)
        self.vol_threshold = self.params.get('vol_threshold', 1.2)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── EMA 趋势 ──
        df['ema_trend'] = df['close'].ewm(span=self.ema_trend, adjust=False).mean()
        df['trend_up'] = (df['close'] > df['ema_trend']).astype(int)

        # ── ATR ──
        prev = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev).abs(),
            (df['low'] - prev).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()

        # ── 成交量 ──
        df['vol_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['vol_ratio'] = (df['volume'] / df['vol_ma']).fillna(1.0)

        # ── Bollinger Bands + 收缩 ──
        bb_mid = df['close'].rolling(self.bb_period, min_periods=1).mean()
        bb_std = df['close'].rolling(self.bb_period, min_periods=1).std()
        df['bb_upper'] = bb_mid + self.bb_std * bb_std
        df['bb_lower'] = bb_mid - self.bb_std * bb_std
        # 带宽 + 收缩
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / bb_mid * 100).fillna(0)
        df['bb_squeeze'] = (
            df['bb_width'].rolling(30, min_periods=20).rank(pct=True) <= 0.25
        ).fillna(0).astype(int)
        # BB 突破
        df['bb_break_up'] = (
            (df['bb_squeeze'] == 1) &
            (df['close'] > df['bb_upper']) &
            (df['close'].shift(1) <= df['bb_upper'].shift(1))
        ).fillna(0).astype(int)
        df['bb_break_dn'] = (
            (df['bb_squeeze'] == 1) &
            (df['close'] < df['bb_lower']) &
            (df['close'].shift(1) >= df['bb_lower'].shift(1))
        ).fillna(0).astype(int)

        # ── Donchian 通道突破 ──
        df['dc_high'] = df['high'].rolling(self.donchian, min_periods=1).max()
        df['dc_low'] = df['low'].rolling(self.donchian, min_periods=1).min()
        df['dc_break_up'] = (
            (df['close'] > df['dc_high'].shift(1))
        ).fillna(0).astype(int)
        df['dc_break_dn'] = (
            (df['close'] < df['dc_low'].shift(1))
        ).fillna(0).astype(int)

        # ── 近期动量（5根K线高低点） ──
        df['high_5'] = df['high'].rolling(5, min_periods=1).max()
        df['low_5'] = df['low'].rolling(5, min_periods=1).min()

        return df

    def on_bar(self, df: pd.DataFrame, idx: int, position: Optional[Position]) -> TradeSetup:
        if position is not None:
            return TradeSetup(action='none')

        row = df.iloc[idx]
        atr = row.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return TradeSetup(action='none')
        if idx < max(self.ema_trend, self.donchian, 30):
            return TradeSetup(action='none')

        close = row['close']
        vol = row.get('vol_ratio', 1.0)
        trend_up = row.get('trend_up', 1)

        # ═══ 做多条件 ═══
        long_signal = False
        long_strength = 0  # 0=无, 1=弱, 2=中, 3=强

        # BB 收缩突破（最强信号）
        if row.get('bb_break_up', 0) == 1 and vol >= self.vol_threshold:
            long_signal = True
            long_strength = max(long_strength, 3)

        # Donchian 突破（中等信号）
        if row.get('dc_break_up', 0) == 1 and vol >= self.vol_threshold:
            long_signal = True
            long_strength = max(long_strength, 2)

        # 5-bar 突破 + 放量（补充信号，需要趋势配合）
        if (close > row.get('high_5', close) and
                vol >= self.vol_threshold * 1.2 and
                trend_up == 1):
            long_signal = True
            long_strength = max(long_strength, 1)

        # 执行做多
        if long_signal and trend_up == 1:
            # 根据信号强度调整止损
            sl_mult = self.stop_mult * (1.2 if long_strength >= 3 else 1.0)
            tp_mult = self.tp_mult * (1.3 if long_strength >= 3 else 1.0)
            stop = close - sl_mult * atr
            tp = close + tp_mult * atr
            return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

        # ═══ 做空条件 ═══
        short_signal = False
        short_strength = 0

        if row.get('bb_break_dn', 0) == 1 and vol >= self.vol_threshold:
            short_signal = True
            short_strength = max(short_strength, 3)

        if row.get('dc_break_dn', 0) == 1 and vol >= self.vol_threshold:
            short_signal = True
            short_strength = max(short_strength, 2)

        if (close < row.get('low_5', close) and
                vol >= self.vol_threshold * 1.2 and
                trend_up == 0):
            short_signal = True
            short_strength = max(short_strength, 1)

        if short_signal and trend_up == 0:
            sl_mult = self.stop_mult * (1.2 if short_strength >= 3 else 1.0)
            tp_mult = self.tp_mult * (1.3 if short_strength >= 3 else 1.0)
            stop = close + sl_mult * atr
            tp = close - tp_mult * atr
            return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        return TradeSetup(action='none')

    def get_trailing_stop(self, position: Position, bar) -> float:
        """阶梯式移动止损。"""
        atr = bar.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return position.stop_loss

        current_stop = position.stop_loss

        if position.type == 'long':
            profit = bar['close'] - position.entry_price
            if profit >= self.trail_start * atr:
                # 阶段1: 保本+
                stage1 = position.entry_price + 0.1 * atr
                trail = bar['close'] - self.trail_tight * atr
                new_stop = max(current_stop, trail, stage1)
                # 阶段2: 盈利超过 50% TP → 更紧
                if profit >= self.tp_mult * atr * 0.5:
                    trail2 = bar['close'] - self.trail_tighter * atr
                    new_stop = max(new_stop, trail2)
                return new_stop
            return current_stop
        else:
            profit = position.entry_price - bar['close']
            if profit >= self.trail_start * atr:
                stage1 = position.entry_price - 0.1 * atr
                trail = bar['close'] + self.trail_tight * atr
                new_stop = min(current_stop, trail, stage1)
                if profit >= self.tp_mult * atr * 0.5:
                    trail2 = bar['close'] + self.trail_tighter * atr
                    new_stop = min(new_stop, trail2)
                return new_stop
            return current_stop
