"""1m 波动扩张剥头皮策略 v2

基于 v1 回测反馈的改进：
  - v1 问题：信号太稀疏（7笔/7天）、杠杆过高导致 -100% 回撤
  - v2 改进：
    1. 去掉波动收缩前置条件（太严格），改为「波动扩张」即时入场
    2. 双重入场：BB突破 + 区间突破（互补信号）
    3. 降低杠杆到 3-5x，控制单笔风险
    4. 更快止盈：ATR×2.0（而非 4.0）
    5. 更紧移动止损：盈利 ATR×0.5 即启动保护

设计依据（近7天 1m 分析）：
  - 峰度 21.1 → 大波动后往往还有余波
  - 平均连续同向 2 根 → 持仓不超过 2-5 分钟
  - ATR×1.0 止损 = 38% 触碰率（可接受）
  - ATR×1.5 止损 = 14% 触碰率（安全）
  - 手续费 0.08% vs ATR×1.5=0.14% → 盈亏比需 >0.6
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..models import TradeSetup, Position


@register('vol_scalp_1m')
class VolScalp1M(Strategy):
    """1m 波动扩张剥头皮。

    Parameters
    ----------
    bb_period : int (20)
    bb_std : float (2.0)
    atr_period : int (14)
    stop_mult : float (1.0)      — 止损 ATR 倍数
    tp_mult : float (2.0)        — 止盈 ATR 倍数（快进快出）
    trail_mult : float (1.0)     — 移动止损 ATR 倍数
    vol_threshold : float (1.5)  — 成交量放大阈值
    range_percentile : float (0.85) — K线波幅分位阈值（>此值视为扩张）
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std = self.params.get('bb_std', 2.0)
        self.atr_period = self.params.get('atr_period', 14)
        self.stop_mult = self.params.get('stop_mult', 1.0)
        self.tp_mult = self.params.get('tp_mult', 2.0)
        self.trail_mult = self.params.get('trail_mult', 1.0)
        self.vol_threshold = self.params.get('vol_threshold', 1.5)
        self.range_percentile = self.params.get('range_percentile', 0.85)

    # ------------------------------------------------------------------
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ---- Bollinger Bands ----
        df['bb_mid'] = df['close'].rolling(window=self.bb_period, min_periods=1).mean()
        bb_std = df['close'].rolling(window=self.bb_period, min_periods=1).std()
        df['bb_upper'] = df['bb_mid'] + self.bb_std * bb_std
        df['bb_lower'] = df['bb_mid'] - self.bb_std * bb_std

        # ---- ATR ----
        prev = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev).abs(),
            (df['low'] - prev).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()

        # ---- K线波幅 ----
        df['bar_range'] = df['high'] - df['low']
        # 波幅的滚动分位数（回看 100 根 ≈ 1.7 小时）
        df['range_90'] = (
            df['bar_range']
            .rolling(window=100, min_periods=30)
            .quantile(self.range_percentile)
        )
        # 当前 K 线是否为扩张 K 线
        df['expansion'] = (
            (df['bar_range'] > df['range_90']).fillna(0).astype(int)
        )

        # ---- 成交量 ----
        df['vol_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['vol_ratio'] = (df['volume'] / df['vol_ma']).fillna(1.0)

        # ---- BB 突破信号 ----
        above_upper = df['close'] > df['bb_upper']
        prev_inside = (df['close'].shift(1) <= df['bb_upper'].shift(1)).fillna(False)
        df['break_up'] = (above_upper & prev_inside).fillna(0).astype(int)

        below_lower = df['close'] < df['bb_lower']
        prev_inside_low = (df['close'].shift(1) >= df['bb_lower'].shift(1)).fillna(False)
        df['break_dn'] = (below_lower & prev_inside_low).fillna(0).astype(int)

        # ---- 近期高低点突破 ----
        df['high_5'] = df['high'].rolling(window=5, min_periods=1).max()
        df['high_10'] = df['high'].rolling(window=10, min_periods=1).max()
        df['low_5'] = df['low'].rolling(window=5, min_periods=1).min()
        df['low_10'] = df['low'].rolling(window=10, min_periods=1).min()

        return df

    # ------------------------------------------------------------------
    def on_bar(self, df: pd.DataFrame, idx: int, position: Optional[Position]) -> TradeSetup:
        if position is not None:
            return TradeSetup(action='none')

        row = df.iloc[idx]

        atr = row.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return TradeSetup(action='none')
        if idx < max(self.bb_period, 100):
            return TradeSetup(action='none')

        close = row['close']
        vol_ratio = row.get('vol_ratio', 1.0)
        expansion = row.get('expansion', 0)
        break_up = row.get('break_up', 0)
        break_dn = row.get('break_dn', 0)

        # ── 信号类型 A：BB 突破 + 放量（核心信号） ──
        if vol_ratio >= self.vol_threshold:
            if break_up == 1:
                stop = close - self.stop_mult * atr
                tp = close + self.tp_mult * atr
                return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

            if break_dn == 1:
                stop = close + self.stop_mult * atr
                tp = close - self.tp_mult * atr
                return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        # ── 信号类型 B：波动扩张 + 方向确认 ──
        if expansion == 1 and vol_ratio >= 1.2:
            # K线收盘在顶部 → 做多突破
            bar_range = row['bar_range']
            if bar_range > 0 and (close - row['low']) / bar_range > 0.6:
                # 同时价格突破 5 周期高点确认
                if close >= row.get('high_5', close):
                    stop = close - self.stop_mult * atr
                    tp = close + self.tp_mult * atr
                    return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

            # K线收盘在底部 → 做空突破
            if bar_range > 0 and (row['high'] - close) / bar_range > 0.6:
                if close <= row.get('low_5', close):
                    stop = close + self.stop_mult * atr
                    tp = close - self.tp_mult * atr
                    return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        return TradeSetup(action='none')

    # ------------------------------------------------------------------
    def get_trailing_stop(self, position: Position, bar) -> float:
        """快速移动止损 — 盈利即保护。"""
        atr = bar.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return position.stop_loss

        current_stop = position.stop_loss

        if position.type == 'long':
            # 只要盈利 > 0，就把止损移到盈亏平衡附近
            if bar['close'] > position.entry_price:
                # 移动止损 = 当前价 - trail_mult×ATR，但不能低于当前止损
                trail_stop = bar['close'] - self.trail_mult * atr
                return max(current_stop, trail_stop, position.entry_price - 0.1 * atr)
            return current_stop
        else:
            if bar['close'] < position.entry_price:
                trail_stop = bar['close'] + self.trail_mult * atr
                return min(current_stop, trail_stop, position.entry_price + 0.1 * atr)
            return current_stop
