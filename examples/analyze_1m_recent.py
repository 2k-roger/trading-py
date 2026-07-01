#!/usr/bin/env python3
"""分析 ETH/USDT 1m 近2天数据 — 市场微观结构、波动率、方向性等"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.data.loader import DataLoader

# 近2天
END_DATE = datetime.now(timezone.utc).strftime('%Y-%m-%d')
START_DATE = (datetime.now(timezone.utc) - timedelta(days=2)).strftime('%Y-%m-%d')

print(f'拉取 ETH/USDT 1m 数据: {START_DATE} → {END_DATE}')
loader = DataLoader('binance')
df = loader.fetch(
    symbol='ETH/USDT',
    timeframe='1m',
    start_date=START_DATE,
    end_date=END_DATE,
    force_download=True,  # 强制更新近2天数据
)

close = df['close']
high = df['high']
low = df['low']
volume = df['volume']

print(f'已加载 {len(df):,} 根K线 ({df.index[0]} → {df.index[-1]})')
print(f'时间跨度: {(df.index[-1] - df.index[0]).total_seconds()/3600:.1f} 小时')
print()

# ================================================================
# 1. 价格概览
# ================================================================
print('=' * 65)
print('📊 价格概览')
print('=' * 65)
print(f'开盘价:      ${close.iloc[0]:.2f}')
print(f'最新价:      ${close.iloc[-1]:.2f}')
print(f'最低价:      ${low.min():.2f}  时间: {low.idxmin()}')
print(f'最高价:      ${high.max():.2f}  时间: {high.idxmax()}')
print(f'均值:        ${close.mean():.2f}')
print(f'中位数:      ${close.median():.2f}')
total_change = (close.iloc[-1] / close.iloc[0] - 1) * 100
print(f'整体涨跌:    {total_change:+.2f}%')
print()

# ================================================================
# 2. 收益率 & 波动率
# ================================================================
print('=' * 65)
print('📈 收益率 & 波动率')
print('=' * 65)
returns = close.pct_change().dropna()
log_returns = np.log(close / close.shift(1)).dropna()

print(f'1分钟均值收益:     {returns.mean()*100:.5f}%')
print(f'1分钟中位数收益:   {returns.median()*100:.5f}%')
print(f'1分钟标准差:       {returns.std()*100:.4f}%')
print(f'年化波动率:        {returns.std() * np.sqrt(365*24*60) * 100:.2f}%')
print(f'偏度:              {returns.skew():.3f}')
print(f'峰度:              {returns.kurtosis():.3f}')
print(f'上涨分钟占比:      {(returns > 0).mean()*100:.1f}%')
print(f'持平分钟占比:      {(returns == 0).mean()*100:.1f}%')
print(f'最大单分钟涨幅:    {returns.max()*100:+.3f}%')
print(f'最大单分钟跌幅:    {returns.min()*100:+.3f}%')
print()

# ================================================================
# 3. 成交量分析
# ================================================================
print('=' * 65)
print('📊 成交量分析')
print('=' * 65)
print(f'总成交量 (ETH):    {volume.sum():,.0f}')
print(f'均笔成交量:        {volume.mean():,.2f}')
print(f'中位数成交量:      {volume.median():,.2f}')
print(f'成交量标准差:      {volume.std():,.2f}')
print(f'最大单分钟量:      {volume.max():,.0f}  时间: {volume.idxmax()}')
# 成交量加权价格
vwap = (close * volume).sum() / volume.sum()
print(f'VWAP (量加权均价): ${vwap:.2f}')
print(f'当前价 vs VWAP:    {(close.iloc[-1]/vwap - 1)*100:+.3f}%')
print()

# ================================================================
# 4. K线形态分析
# ================================================================
print('=' * 65)
print('🕯️ K线形态分析')
print('=' * 65)
body = (close - df['open']).abs()
upper_wick = high - df[['open', 'close']].max(axis=1)
lower_wick = df[['open', 'close']].min(axis=1) - low
total_range = high - low

print(f'平均实体:          ${body.mean():.3f} ({ (body/close*100).mean():.3f}%)')
print(f'平均上影线:        ${upper_wick.mean():.3f} ({ (upper_wick/close*100).mean():.3f}%)')
print(f'平均下影线:        ${lower_wick.mean():.3f} ({ (lower_wick/close*100).mean():.3f}%)')
print(f'平均总波幅:        ${total_range.mean():.3f} ({ (total_range/close*100).mean():.3f}%)')
print(f'最大波幅:          ${total_range.max():.3f} ({ (total_range/close*100).max():.3f}%)')
print(f'最大波幅时间:      {total_range.idxmax()}')

# 实体占比
body_pct = body / (total_range + 1e-10)
print(f'实体占波幅比 (均值): {body_pct.mean()*100:.1f}%')
print(f'  (>70% 强趋势):     {(body_pct > 0.7).mean()*100:.1f}%')
print(f'  (<30% 十字星):     {(body_pct < 0.3).mean()*100:.1f}%')

# 阳线/阴线
green = close > df['open']
red = close < df['open']
doji = close == df['open']
print(f'阳线: {green.sum()} ({green.mean()*100:.1f}%)')
print(f'阴线: {red.sum()} ({red.mean()*100:.1f}%)')
print(f'十字: {doji.sum()} ({doji.mean()*100:.1f}%)')
print()

# ================================================================
# 5. 波动率分时段
# ================================================================
print('=' * 65)
print('⏰ 分时段波动率 (按UTC小时)')
print('=' * 65)
hourly_std = returns.groupby(returns.index.hour).std() * 100
hourly_count = returns.groupby(returns.index.hour).count()
for h in range(24):
    if h in hourly_std.index:
        s = hourly_std[h]
        n = hourly_count[h]
        bar = '█' * int(s / hourly_std.max() * 30) if hourly_std.max() > 0 else ''
        print(f'  {h:02d}:00 UTC | {s:.4f}% std | n={n:3d} | {bar}')
    else:
        print(f'  {h:02d}:00 UTC | (无数据)')
print()

# ================================================================
# 6. 大波动检测
# ================================================================
print('=' * 65)
print('⚡ 极端波动事件 (收益率 > 3σ)')
print('=' * 65)
threshold = returns.std() * 3
spikes = returns[returns.abs() > threshold].sort_values()
if len(spikes) > 0:
    for ts, r in spikes.items():
        direction = '📈' if r > 0 else '📉'
        print(f'  {direction} {ts} | {r*100:+.3f}% | 价格: ${close[ts]:.2f}')
    print(f'共 {len(spikes)} 次极端波动')
else:
    print('  无极端波动事件')

# 最大波动TOP10
print()
print('最大波动 TOP10:')
top10 = returns.abs().nlargest(10)
for ts in top10.index:
    r = returns[ts] * 100
    direction = '📈' if r > 0 else '📉'
    print(f'  {direction} {ts} | {r:+.3f}% | 价格: ${close[ts]:.2f} | 量: {volume[ts]:,.0f}')
print()

# ================================================================
# 7. 回撤分析
# ================================================================
print('=' * 65)
print('📉 最大回撤 (近2天)')
print('=' * 65)
peak = close.expanding().max()
drawdown = (close - peak) / peak * 100
max_dd = drawdown.min()
max_dd_time = drawdown.idxmin()
print(f'最大回撤:           {max_dd:.3f}%')
print(f'最大回撤时间:        {max_dd_time}')
# Find recovery
if max_dd < 0:
    after_dd = close[max_dd_time:]
    recovery = (after_dd >= peak[max_dd_time])
    if recovery.any():
        recovery_time = recovery.idxmax()
        dur = (recovery_time - max_dd_time).total_seconds() / 60
        print(f'恢复时间:            {recovery_time} (耗时 {dur:.0f} 分钟)')
    else:
        print(f'尚未恢复')
print()

# ================================================================
# 8. 趋势与方向统计
# ================================================================
print('=' * 65)
print('📐 趋势分析')
print('=' * 65)
# 线性趋势
x = np.arange(len(close))
slope, intercept = np.polyfit(x, close.values, 1)
slope_hour = slope * 60
print(f'线性趋势斜率:        ${slope:.6f}/分钟 (${slope_hour:.4f}/小时)')

# EMA
ema_20 = close.ewm(span=20, adjust=False).mean()
ema_50 = close.ewm(span=50, adjust=False).mean()
ema_200 = close.ewm(span=200, adjust=False).mean()
print(f'当前价 vs EMA20:     {(close.iloc[-1]/ema_20.iloc[-1] - 1)*100:+.3f}%')
print(f'当前价 vs EMA50:     {(close.iloc[-1]/ema_50.iloc[-1] - 1)*100:+.3f}%')
print(f'当前价 vs EMA200:    {(close.iloc[-1]/ema_200.iloc[-1] - 1)*100:+.3f}%')

# 连续方向
direction = (returns > 0).astype(int)
runs = []
current_run = 1
for i in range(1, len(direction)):
    if direction.iloc[i] == direction.iloc[i-1]:
        current_run += 1
    else:
        runs.append(current_run)
        current_run = 1
runs.append(current_run)
runs = pd.Series(runs)
print(f'最长连续同向K线:     {runs.max()} 分钟')
print(f'平均连续同向:        {runs.mean():.1f} 分钟')
print()

# ================================================================
# 9. 支撑阻力 (简单 pivot)
# ================================================================
print('=' * 65)
print('📍 近期支撑/阻力 (基于局部高低点)')
print('=' * 65)
# 简单 pivot high/low (窗口=20)
window = 20
pivot_high = high[(high.rolling(window, center=True).max() == high)]
pivot_low = low[(low.rolling(window, center=True).min() == low)]

# 最近5个
recent_highs = pivot_high.tail(5)
recent_lows = pivot_low.tail(5)

print('近期阻力位:')
for ts, p in recent_highs.items():
    print(f'  ${p:.2f} ({ts})')
print('近期支撑位:')
for ts, p in recent_lows.items():
    print(f'  ${p:.2f} ({ts})')
print()

# ================================================================
# 10. 策略启发
# ================================================================
print('=' * 65)
print('💡 策略启发')
print('=' * 65)

# 最优止损距离的经验估计
tr = pd.concat([
    high - low,
    (high - close.shift(1)).abs(),
    (low - close.shift(1)).abs(),
], axis=1).max(axis=1)
atr_14 = tr.ewm(alpha=1.0/14, adjust=False).mean()
atr_pct = atr_14 / close * 100

print(f'ATR(14):             ${atr_14.iloc[-1]:.2f} ({atr_pct.iloc[-1]:.3f}%)')
print(f'ATR(14) 均值:        ${atr_14.mean():.2f} ({atr_pct.mean():.3f}%)')
print(f'ATR(14) 中位数:      ${atr_14.median():.2f} ({atr_pct.median():.3f}%)')

# 不同倍数止损被触碰概率
for mult in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    stop_dist = atr_14 * mult
    # 简单估计: 单根K线波幅超过止损距离的概率
    hit_pct = (total_range > stop_dist).mean() * 100
    hit_in_5 = (total_range.rolling(5).max() > stop_dist).mean() * 100
    print(f'  ATR×{mult:.2f}: 止损=${stop_dist.iloc[-1]:.3f} '
          f'({(stop_dist.iloc[-1]/close.iloc[-1]*100):.3f}%) | '
          f'1K线触碰={hit_pct:.1f}% | 5K线触碰={hit_in_5:.1f}%')

print()
print(f'💡 1m K线共 {len(df):,} 根，建议重点关注：')
print(f'   - 波动率最高的时段（避开或利用）')
print(f'   - ATR×0.25~0.5 紧止损是否可行')
print(f'   - 大波动事件是否可捕捉（突破策略）')
