"""Volatility Breakout Strategy — 专为 1m 周期设计。

设计依据（近7天 1m 数据分析）：
  - 噪声率 52%, ACF≈0, VR≈1 → 价格近似随机游走，不可预测方向
  - 峰度 21.1（远超正态 3.0）→ 肥尾，极端波动频繁
  - ATR=0.096%, 均波幅=0.096%, 90分位=0.193%
  - 大波动密度: 4.8次/100根K线

策略逻辑：
  1. 计算 Bollinger Bands (20,2) — 识别波动率收缩/扩张
  2. 当带宽（BandWidth = (upper-lower)/mid）处于 N 周期低位 → 波动收缩
  3. 收缩后价格突破区间 + 成交量确认 → 入场
  4. 紧止损 + 肥尾止盈：ATR×1.5 止损, ATR×4.0 止盈
  5. ATR 移动止损：价格朝有利方向移动超过 1×ATR 后启动

不设方向偏好 — 做多/做空对称。
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..models import TradeSetup, Position


@register('vol_breakout_1m')
class VolatilityBreakout1M(Strategy):
    """1m 波动率突破策略 — 捕捉肥尾事件。

    Parameters
    ----------
    bb_period : int (default 20)
        Bollinger Band 计算周期。
    bb_std : float (default 2.0)
        Bollinger Band 标准差倍数。
    squeeze_period : int (default 10)
        波动收缩回看周期：带宽必须在 squeeze_period 内处于低位。
    squeeze_percentile : float (default 0.2)
        带宽低位的分位数阈值（0.2 = 带宽在历史底部 20% 才触发）。
    volume_trigger : float (default 1.5)
        成交量放大倍数（相对于 20 周期均值）。
    atr_period : int (default 14)
        ATR 计算周期。
    stop_mult : float (default 1.5)
        止损 = ATR × stop_mult。
    tp_mult : float (default 4.0)
        止盈 = ATR × tp_mult（利用肥尾）。
    trail_trigger_mult : float (default 1.0)
        价格朝有利方向移动超过 ATR×trail_trigger 后，启动移动止损。
    trail_mult : float (default 1.5)
        移动止损距离 = ATR × trail_mult。
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std = self.params.get('bb_std', 2.0)
        self.squeeze_period = self.params.get('squeeze_period', 10)
        self.squeeze_percentile = self.params.get('squeeze_percentile', 0.2)
        self.volume_trigger = self.params.get('volume_trigger', 1.5)
        self.atr_period = self.params.get('atr_period', 14)
        self.stop_mult = self.params.get('stop_mult', 1.5)
        self.tp_mult = self.params.get('tp_mult', 4.0)
        self.trail_trigger_mult = self.params.get('trail_trigger_mult', 1.0)
        self.trail_mult = self.params.get('trail_mult', 1.5)

    # ------------------------------------------------------------------
    # Strategy API
    # ------------------------------------------------------------------

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 Bollinger Bands、波动收缩信号、成交量比、ATR。

        所有信号列保证无 NaN：fillna(0) 处理 early bars 的缺失值。
        """
        df = df.copy()

        # ---- Bollinger Bands ----
        df['bb_mid'] = df['close'].rolling(window=self.bb_period, min_periods=1).mean()
        bb_std = df['close'].rolling(window=self.bb_period, min_periods=1).std()
        df['bb_upper'] = df['bb_mid'] + self.bb_std * bb_std
        df['bb_lower'] = df['bb_mid'] - self.bb_std * bb_std

        # 带宽 = (upper - lower) / mid * 100
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

        # 波动收缩信号（向量化）：
        # 用 rolling rank(pct) 替代慢速 apply(lambda quantile)
        # rank(pct=True) 返回当前值在窗口中的分位 (0~1)
        # 分位越低 = 带宽越窄 = 收缩
        bb_width_pct_rank = (
            df['bb_width']
            .rolling(window=self.squeeze_period, min_periods=self.squeeze_period)
            .rank(pct=True)
        )
        df['squeeze'] = (
            (bb_width_pct_rank <= self.squeeze_percentile)
            .fillna(0)
            .astype(int)
        )

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

        # ---- 突破信号 (fillna 处理 early bars) ----
        # 向上突破：收盘 > BB上轨 且 前一根在轨内
        above_upper = df['close'] > df['bb_upper']
        prev_inside = (df['close'].shift(1) <= df['bb_upper'].shift(1)).fillna(False)
        df['break_up'] = (above_upper & prev_inside).fillna(0).astype(int)

        # 向下突破：收盘 < BB下轨 且 前一根在轨内
        below_lower = df['close'] < df['bb_lower']
        prev_inside_low = (df['close'].shift(1) >= df['bb_lower'].shift(1)).fillna(False)
        df['break_dn'] = (below_lower & prev_inside_low).fillna(0).astype(int)

        # ---- 近期高低点（回看5根）----
        df['high_5'] = df['high'].rolling(window=5, min_periods=1).max()
        df['low_5'] = df['low'].rolling(window=5, min_periods=1).min()

        return df

    def on_bar(
        self,
        df: pd.DataFrame,
        idx: int,
        position: Optional[Position],
    ) -> TradeSetup:
        if position is not None:
            return TradeSetup(action='none')

        row = df.iloc[idx]

        # 前置条件：指标必须就绪
        atr = row.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return TradeSetup(action='none')

        # 需要足够的 lookback
        if idx < self.bb_period + self.squeeze_period:
            return TradeSetup(action='none')

        close = row['close']
        vol_ratio = row.get('vol_ratio', 1.0)
        squeeze = row.get('squeeze', 0)
        break_up = row.get('break_up', 0)
        break_dn = row.get('break_dn', 0)

        # ---- 入场条件 ----
        # 1) 波动收缩信号
        # 2) 价格突破 BB 轨道
        # 3) 成交量确认（放量 > 阈值）
        # 4) 价格在近期区间外确认

        if squeeze == 1 and vol_ratio >= self.volume_trigger:
            # 向上突破 → 做多
            if break_up == 1:
                stop = close - self.stop_mult * atr
                tp = close + self.tp_mult * atr
                return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

            # 向下突破 → 做空
            if break_dn == 1:
                stop = close + self.stop_mult * atr
                tp = close - self.tp_mult * atr
                return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        # ---- 备用信号：强动量突破（无收缩要求，但需要更大成交量） ----
        # 宽松版：放量突破近期区间 + 极高成交量
        if vol_ratio >= self.volume_trigger * 1.5:
            high_5 = row.get('high_5', np.nan)
            low_5 = row.get('low_5', np.nan)

            if not np.isnan(high_5) and close > high_5:
                stop = close - self.stop_mult * atr
                tp = close + self.tp_mult * atr
                return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

            if not np.isnan(low_5) and close < low_5:
                stop = close + self.stop_mult * atr
                tp = close - self.tp_mult * atr
                return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        return TradeSetup(action='none')

    # ------------------------------------------------------------------
    # 移动止损
    # ------------------------------------------------------------------

    def get_trailing_stop(self, position: Position, bar) -> float:
        """ATR 移动止损 — 价格朝有利方向移动超过触发距离后启动。"""
        atr = bar.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return position.stop_loss

        current_stop = position.stop_loss
        trigger_dist = self.trail_trigger_mult * atr

        if position.type == 'long':
            # 盈利超过 trigger 后，止损上移到 当前价 - trail_mult×ATR
            profit_dist = bar['close'] - position.entry_price
            if profit_dist >= trigger_dist:
                trail_stop = bar['close'] - self.trail_mult * atr
                return max(current_stop, trail_stop)
            return current_stop
        else:
            # 做空
            profit_dist = position.entry_price - bar['close']
            if profit_dist >= trigger_dist:
                trail_stop = bar['close'] + self.trail_mult * atr
                return min(current_stop, trail_stop)
            return current_stop
