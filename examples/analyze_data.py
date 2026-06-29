#!/usr/bin/env python3
"""Analyze ETH/USDT data to understand market conditions and inform strategy design."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.data.loader import DataLoader

# Fetch data
loader = DataLoader('binance')
df = loader.fetch(
    symbol='ETH/USDT',
    timeframe='1h',
    start_date='2026-04-01',
    end_date='2026-06-25',
)

close = df['close']
high = df['high']
low = df['low']

print(f'Loaded {len(df):,} bars ({df.index[0]} → {df.index[-1]})')
print(f'Period: {(df.index[-1] - df.index[0]).days} days')
print()

# ---- 1. Price overview ----
print('=' * 60)
print('PRICE OVERVIEW')
print('=' * 60)
print(f'Open price:  ${close.iloc[0]:.2f}')
print(f'Close price: ${close.iloc[-1]:.2f}')
print(f'Min:         ${low.min():.2f}  (on {low.idxmin()})')
print(f'Max:         ${high.max():.2f}  (on {high.idxmax()})')
print(f'Mean close:  ${close.mean():.2f}')
print(f'Median:      ${close.median():.2f}')
print()

# ---- 2. Returns & volatility ----
print('=' * 60)
print('RETURNS & VOLATILITY')
print('=' * 60)
returns = close.pct_change().dropna()
log_returns = np.log(close / close.shift(1)).dropna()

print(f'Hourly mean return:     {returns.mean()*100:.4f}%')
print(f'Hourly median return:   {returns.median()*100:.4f}%')
print(f'Hourly std:             {returns.std()*100:.4f}%')
print(f'Hourly std (log):       {log_returns.std()*100:.4f}%')
print(f'Annualized vol:         {returns.std() * np.sqrt(365*24) * 100:.2f}%')
print(f'Skewness:               {returns.skew():.3f}')
print(f'Kurtosis:               {returns.kurtosis():.3f}')
print(f'% positive hours:       {(returns > 0).mean()*100:.1f}%')
print(f'Max 1h gain:            {returns.max()*100:+.2f}%')
print(f'Max 1h drop:            {returns.min()*100:+.2f}%')
print()

# ---- 3. Drawdown analysis ----
print('=' * 60)
print('DRAWDOWN ANALYSIS (buy & hold)')
print('=' * 60)
peak = close.expanding().max()
drawdown = (close - peak) / peak * 100
max_dd = drawdown.min()
max_dd_date = drawdown.idxmin()
print(f'Max drawdown:           {max_dd:.2f}% (on {max_dd_date})')

# Recovery
dd_periods = []
in_dd = False
dd_start = None
for i in range(len(drawdown)):
    if drawdown.iloc[i] < -1 and not in_dd:
        in_dd = True
        dd_start = drawdown.index[i]
    elif drawdown.iloc[i] > -0.5 and in_dd:
        dd_periods.append((dd_start, drawdown.index[i], (drawdown.index[i] - dd_start).total_seconds() / 3600))
        in_dd = False

if dd_periods:
    durations = [d[2] for d in dd_periods]
    print(f'Number of DD episodes:  {len(dd_periods)}')
    print(f'Avg DD duration:        {np.mean(durations):.1f} hours')
    print(f'Max DD duration:        {np.max(durations):.1f} hours')
print()

# ---- 4. Trend analysis ----
print('=' * 60)
print('TREND ANALYSIS')
print('=' * 60)
# Linear regression slope
x = np.arange(len(close))
slope, intercept = np.polyfit(x, close.values, 1)
print(f'Linear trend slope:     ${slope:.4f}/hour (${slope*24*30:.2f}/month)')

# Up/down periods
up_days = (close.resample('D').last().diff() > 0).sum()
down_days = (close.resample('D').last().diff() < 0).sum()
print(f'Up days:                {up_days}')
print(f'Down days:              {down_days}')
print(f'Up/down ratio:          {up_days/max(down_days,1):.2f}')

# EMA trend bias
ema_50 = close.ewm(span=50, adjust=False).mean()
ema_200 = close.ewm(span=200, adjust=False).mean()
pct_above_50 = (close > ema_50).mean() * 100
pct_above_200 = (close > ema_200).mean() * 100
print(f'% time above EMA50:     {pct_above_50:.1f}%')
print(f'% time above EMA200:    {pct_above_200:.1f}%')
print()

# ---- 5. Volatility regime ----
print('=' * 60)
print('VOLATILITY REGIME (ATR)')
print('=' * 60)
prev_close = close.shift(1)
tr = pd.concat([
    high - low,
    (high - prev_close).abs(),
    (low - prev_close).abs(),
], axis=1).max(axis=1)
atr_14 = tr.ewm(alpha=1.0/14, adjust=False).mean()
atr_pct = atr_14 / close * 100

print(f'ATR(14) mean:           ${atr_14.mean():.2f} ({atr_pct.mean():.3f}%)')
print(f'ATR(14) median:         ${atr_14.median():.2f} ({atr_pct.median():.3f}%)')
print(f'ATR(14) 90th pct:       ${atr_14.quantile(0.9):.2f} ({atr_pct.quantile(0.9):.3f}%)')
print(f'ATR(14) min/max:        ${atr_14.min():.2f} / ${atr_14.max():.2f}')
print()

# ---- 6. Liquidation distance analysis (100x) ----
print('=' * 60)
print('LIQUIDATION DISTANCE (100x leverage)')
print('=' * 60)
lev = 100
mmr = 0.005
liq_dist = (1 - (1 - 1/lev + mmr)) * 100  # for longs
print(f'Liquidation distance:   {liq_dist:.4f}% (below entry)')
print(f'ATR(14)*3 mean stop:    {atr_pct.mean()*3:.4f}% (below entry)')
print(f'Ratio ATR3/liq_dist:    {atr_pct.mean()*3/liq_dist:.1f}x')
print()

# Probability of touching liquidation vs stop
print('What fraction of 1h bars would trigger liquidation vs stop?')
n_bars = len(df)
liq_triggers = 0
stop_triggers = 0
for i in range(1, n_bars):
    entry = close.iloc[i-1]
    liq_price = entry * (1 - 1/lev + mmr)
    stop_price = entry - atr_14.iloc[i-1] * 3
    # Check if next bar touches either
    if low.iloc[i] <= liq_price:
        liq_triggers += 1
    if low.iloc[i] <= stop_price:
        stop_triggers += 1

print(f'Bars touching liq first:  {liq_triggers} ({liq_triggers/n_bars*100:.1f}%)')
print(f'Bars touching stop first: {stop_triggers} ({stop_triggers/n_bars*100:.1f}%)')
print()

# ---- 7. Mean reversion analysis ----
print('=' * 60)
print('MEAN REVERSION')
print('=' * 60)
# Autocorrelation of returns
for lag in [1, 2, 4, 8, 24]:
    autocorr = returns.autocorr(lag=lag)
    print(f'Return autocorr (lag={lag:2d}h):  {autocorr:+.4f}')

# After big moves, what happens?
big_up = returns > returns.std() * 2
big_down = returns < -returns.std() * 2
if big_up.any():
    next_after_up = returns.shift(-1)[big_up].dropna()
    print(f'After +2σ up: mean next hour = {next_after_up.mean()*100:+.4f}% (N={len(next_after_up)})')
if big_down.any():
    next_after_down = returns.shift(-1)[big_down].dropna()
    print(f'After -2σ down: mean next hour = {next_after_down.mean()*100:+.4f}% (N={len(next_after_down)})')
print()

# ---- 8. Range analysis ----
print('=' * 60)
print('BAR RANGE ANALYSIS')
print('=' * 60)
bar_range_pct = (high - low) / close * 100
print(f'Avg bar range:          {bar_range_pct.mean():.3f}%')
print(f'Median bar range:       {bar_range_pct.median():.3f}%')
print(f'90th pct bar range:     {bar_range_pct.quantile(0.9):.3f}%')
print(f'Max bar range:          {bar_range_pct.max():.3f}%')

# Key insight: at 100x, a 0.5% move liquidates — what fraction of bars have range > 0.5%?
liq_pct = liq_dist
pct_bars_kill = (bar_range_pct > liq_pct).mean() * 100
print(f'% bars with range > {liq_pct:.3f}% (lethal):  {pct_bars_kill:.1f}%')
print()

# ---- 9. Optimal stop distance (empirical) ----
print('=' * 60)
print('OPTIMAL STOP DISTANCE (empirical)')
print('=' * 60)
for mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
    stop_dist_pct = atr_pct * mult
    pct_hit = (stop_dist_pct > liq_pct).mean() * 100
    avg_stop = stop_dist_pct.mean()
    print(f'  ATR×{mult:.2f}: avg stop = {avg_stop:.3f}%, stop > liq in {pct_hit:.1f}% of bars')

print()

# ---- 10. Consecutive directional moves ----
print('=' * 60)
print('CONSECUTIVE DIRECTIONAL MOVES')
print('=' * 60)
direction = (returns > 0).astype(int)
# Find runs
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
print(f'Max consecutive same-direction bars: {runs.max()}')
print(f'Mean consecutive:                    {runs.mean():.1f}')
print(f'Median consecutive:                  {runs.median():.1f}')
print(f'90th percentile:                     {runs.quantile(0.9):.1f}')
print()

# ---- 11. Hour-of-day seasonality ----
print('=' * 60)
print('HOURLY SEASONALITY')
print('=' * 60)
hourly_returns = returns.groupby(returns.index.hour).mean() * 100
best_hour = hourly_returns.idxmax()
worst_hour = hourly_returns.idxmin()
print(f'Best hour:  {best_hour:02d}:00 ({hourly_returns[best_hour]:+.4f}% avg)')
print(f'Worst hour: {worst_hour:02d}:00 ({hourly_returns[worst_hour]:+.4f}% avg)')
print(f'Best 3 hours: {hourly_returns.nlargest(3).index.tolist()}')
print(f'Worst 3 hours: {hourly_returns.nsmallest(3).index.tolist()}')

# Day-of-week seasonality
dow_returns = returns.groupby(returns.index.dayofweek).mean() * 100
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for d, r in zip(dow_names, dow_returns):
    print(f'  {d}: {r:+.4f}%')
print()

# ---- 12. Key takeaways for strategy design ----
print('=' * 60)
print('KEY TAKEAWAYS FOR STRATEGY DESIGN')
print('=' * 60)
print(f'1. At 100x, liquidation is only {liq_dist:.2f}% away — extremely tight')
print(f'2. ATR(14)×3 stop ({atr_pct.mean()*3:.2f}% avg) is MUCH wider than liq distance')
print(f'3. Need either: lower leverage, MUCH tighter stops, or different approach')
print(f'4. {pct_bars_kill:.0f}% of 1h bars have range large enough to trigger liquidation')
