"""15m 均值回复策略 — 基于 VR=0.932 的统计结构。

设计依据（近7天 15m 分析）：
  - VR(5)=0.922, VR(10)=0.932, VR(20)=0.889 → 所有窗口 VR<1，明确均值回复
  - ACF(1)=-0.002 → 弱负自相关，符合均值回复特征
  - ATR=0.456%，手续费仅占 ATR×1.5 的 12% → 充裕利润空间
  - ATR×1.5 止损触碰率 12.5% → 合理止损距离
  - 峰度 18.5 → 极端偏离后回归更强

策略逻辑：
  1. 均值锚点：EMA(20) 作为短期公允价值参考
  2. 偏离度量：BB(20,2) + RSI(14) 双确认
  3. 趋势过滤：ADX > 25 时禁止反向交易（不在强趋势中接飞刀）
  4. 入场条件（需同时满足）：
     - 价格触及 BB 外轨（上轨=做空，下轨=做多）
     - RSI 极端（>65 做空，<35 做多）
     - ADX < 25（非强趋势环境）
     - 成交量 ≥ 1.0×均值（非冷清行情）
  5. 止盈：回归 BB 中轨（EMA20）
  6. 止损：ATR×1.5
  7. 时间止损：持仓超 12 根K线未回归 → 强制退出
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..models import TradeSetup, Position


@register('mean_rev_15m')
class MeanRev15M(Strategy):
    """15m Bollinger Band 均值回复 + RSI 确认 + ADX 趋势过滤。

    Parameters
    ----------
    bb_period : int (20)
    bb_std : float (2.0)
    rsi_period : int (14)
    rsi_oversold : int (35)   — RSI 低于此值考虑做多
    rsi_overbought : int (65) — RSI 高于此值考虑做空
    adx_period : int (14)
    adx_max : int (25)        — ADX 超过此值禁止入场（强趋势中不接飞刀）
    ema_period : int (20)     — 均值锚点
    atr_period : int (14)
    stop_mult : float (1.5)   — 止损 ATR 倍数
    vol_min : float (1.0)     — 最小成交量比（相对于20周期均值）
    max_hold_bars : int (12)  — 时间止损（3小时）
    """

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std = self.params.get('bb_std', 2.0)
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_oversold = self.params.get('rsi_oversold', 35)
        self.rsi_overbought = self.params.get('rsi_overbought', 65)
        self.adx_period = self.params.get('adx_period', 14)
        self.adx_max = self.params.get('adx_max', 25)
        self.ema_period = self.params.get('ema_period', 20)
        self.atr_period = self.params.get('atr_period', 14)
        self.stop_mult = self.params.get('stop_mult', 1.5)
        self.vol_min = self.params.get('vol_min', 1.0)
        self.max_hold_bars = self.params.get('max_hold_bars', 12)
        self.cooldown_bars = self.params.get('cooldown_bars', 5)
        # 防止瀑布行情中连续抄底/摸顶
        self._last_signal_idx = -999
        self._last_signal_dir = None

    # ------------------------------------------------------------------
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ---- Bollinger Bands ----
        df['bb_mid'] = df['close'].ewm(span=self.bb_period, adjust=False).mean()
        bb_std = df['close'].rolling(window=self.bb_period, min_periods=1).std()
        df['bb_upper'] = df['bb_mid'] + self.bb_std * bb_std
        df['bb_lower'] = df['bb_mid'] - self.bb_std * bb_std

        # %B = (price - lower) / (upper - lower), 0=在下轨, 1=在上轨
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_pct_b'] = ((df['close'] - df['bb_lower']) / bb_range).fillna(0.5)

        # 偏离距离 (ATR 标准化)
        df['dist_from_mid'] = (df['close'] - df['bb_mid']) / df['bb_mid'] * 100

        # ---- RSI ----
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
        df['rsi'] = df['rsi'].fillna(50.0)

        # ---- ATR ----
        prev = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev).abs(),
            (df['low'] - prev).abs(),
        ], axis=1).max(axis=1)
        # 用 SMA 初始化再用 EMA（避免前期 NaN 累积）
        atr_sma = tr.rolling(window=self.atr_period, min_periods=1).mean()
        df['atr'] = tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()
        # 前 N 根用 SMA 填充
        df['atr'] = df['atr'].fillna(atr_sma)

        # ---- ADX (趋势强度) ----
        up_move = df['high'].diff().fillna(0)
        down_move = -df['low'].diff().fillna(0)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smooth with Wilder's method (EMA alpha=1/N)
        atr_nz = df['atr'].clip(lower=1e-10)  # avoid div by zero
        smooth_plus_dm = pd.Series(plus_dm).ewm(alpha=1.0 / self.adx_period, adjust=False).mean()
        smooth_minus_dm = pd.Series(minus_dm).ewm(alpha=1.0 / self.adx_period, adjust=False).mean()
        plus_di = (100.0 * smooth_plus_dm / atr_nz).fillna(0).clip(0, 100)
        minus_di = (100.0 * smooth_minus_dm / atr_nz).fillna(0).clip(0, 100)

        di_sum = (plus_di + minus_di).clip(lower=1e-10)
        dx = (abs(plus_di - minus_di) / di_sum * 100.0).fillna(0)
        df['adx'] = dx.ewm(alpha=1.0 / self.adx_period, adjust=False).mean().fillna(0).clip(0, 100)
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # ---- 成交量 ----
        df['vol_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['vol_ratio'] = (df['volume'] / df['vol_ma']).fillna(1.0)

        # ---- 超卖/超买信号 ----
        # 做多信号：价格跌破下轨 + RSI 超卖 + ADX 不高 + 有量
        df['long_sig'] = (
            (df['close'] < df['bb_lower']) &
            (df['rsi'] < self.rsi_oversold) &
            (df['adx'] < self.adx_max) &
            (df['vol_ratio'] >= self.vol_min)
        ).fillna(0).astype(int)

        # 做空信号：价格突破上轨 + RSI 超买 + ADX 不高 + 有量
        df['short_sig'] = (
            (df['close'] > df['bb_upper']) &
            (df['rsi'] > self.rsi_overbought) &
            (df['adx'] < self.adx_max) &
            (df['vol_ratio'] >= self.vol_min)
        ).fillna(0).astype(int)

        # ---- 增强信号：双重极端（可选） ----
        # 距离均值超过 2×ATR → 更强的回归信号
        df['extreme_dist'] = (
            (df['close'] - df['bb_mid']).abs() > df['atr'] * 2.5
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
        if idx < max(self.bb_period + self.adx_period, 30):
            return TradeSetup(action='none')

        close = row['close']
        bb_mid = row['bb_mid']
        rsi = row.get('rsi', 50)
        adx = row.get('adx', 20)
        vol_ratio = row.get('vol_ratio', 1.0)
        extreme = row.get('extreme_dist', 0)

        # ═══ 做多（超卖回归） ═══
        # 入场条件：价格在 BB 下轨附近 + RSI 超卖 + 非强趋势
        long_signal = row.get('long_sig', 0)

        # 备选入场：价格虽未触及下轨但极端偏离均值 + RSI 接近超卖
        long_alt = (
            extreme == 1
            and rsi < self.rsi_oversold + 5  # 略宽松
            and adx < self.adx_max
            and vol_ratio >= self.vol_min
            and close < bb_mid  # 至少低于均值
        )

        if long_signal or long_alt:
            # 冷却检查：同方向信号需间隔 cooldown_bars
            if (self._last_signal_dir == 'long'
                    and idx - self._last_signal_idx < self.cooldown_bars):
                return TradeSetup(action='none')

            # 瀑布保护：做多时，确保近3根K线未创新低
            recent_low = df['low'].iloc[max(0, idx-3):idx+1].min()
            if close <= recent_low * 1.001:  # 仍在创新低→不接
                return TradeSetup(action='none')

            self._last_signal_idx = idx
            self._last_signal_dir = 'long'

            stop = close - self.stop_mult * atr
            # 止盈 = 回归均值（稍微保守一点，到均线附近）
            tp = bb_mid
            # 确保止盈 > 入场价（做多）
            if tp <= close:
                tp = close + self.stop_mult * atr * 1.2  # 退而求其次
            return TradeSetup(action='enter_long', stop_loss=stop, take_profit=tp)

        # ═══ 做空（超买回归） ═══
        short_signal = row.get('short_sig', 0)

        short_alt = (
            extreme == 1
            and rsi > self.rsi_overbought - 5
            and adx < self.adx_max
            and vol_ratio >= self.vol_min
            and close > bb_mid
        )

        if short_signal or short_alt:
            if (self._last_signal_dir == 'short'
                    and idx - self._last_signal_idx < self.cooldown_bars):
                return TradeSetup(action='none')

            # 火箭保护：做空时，确保近3根K线未创新高
            recent_high = df['high'].iloc[max(0, idx-3):idx+1].max()
            if close >= recent_high * 0.999:
                return TradeSetup(action='none')

            self._last_signal_idx = idx
            self._last_signal_dir = 'short'

            stop = close + self.stop_mult * atr
            tp = bb_mid
            if tp >= close:
                tp = close - self.stop_mult * atr * 1.2
            return TradeSetup(action='enter_short', stop_loss=stop, take_profit=tp)

        return TradeSetup(action='none')

    # ------------------------------------------------------------------
    def get_trailing_stop(self, position: Position, bar) -> float:
        """均值回复策略的止损管理。

        均值回复的核心信仰：价格会回归均值。
        - 不做激进移动止损（会让微利截断大回归）
        - 只有当价格已完成大部分回归（>70% 距离均值），才收紧止损保本
        - 否则让固定止损和止盈各司其职
        """
        atr = bar.get('atr', np.nan)
        if np.isnan(atr) or atr <= 0:
            return position.stop_loss

        current_stop = position.stop_loss
        bb_mid = bar.get('bb_mid', np.nan)

        # 需要 BB 中轨来判断回归进度
        if np.isnan(bb_mid):
            return current_stop

        if position.type == 'long':
            # 计算回归进度：入场时在低位，目标是 bb_mid
            total_dist = bb_mid - position.entry_price
            current_progress = bar['close'] - position.entry_price

            # 只有当完成 >70% 回归 且 已盈利 >0，才保本
            if total_dist > 0 and current_progress > total_dist * 0.7 and bar['close'] > position.entry_price:
                breakeven = position.entry_price + 0.05 * atr
                return max(current_stop, breakeven)
            return current_stop
        else:
            total_dist = position.entry_price - bb_mid
            current_progress = position.entry_price - bar['close']

            if total_dist > 0 and current_progress > total_dist * 0.7 and bar['close'] < position.entry_price:
                breakeven = position.entry_price - 0.05 * atr
                return min(current_stop, breakeven)
            return current_stop
