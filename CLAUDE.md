# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (Python ≥ 3.10)
pip install -r requirements.txt

# Run backtest (primary entry point)
python examples/run_backtest.py

# Live trading via WebSocket (Binance)
python examples/live_trading.py
python examples/live_trading.py --log-level DEBUG

# Grid-search parameter optimization (~200-300 backtests, writes to data/opt_results.json)
python examples/optimize.py

# Market data analysis
python examples/analyze_data.py

# Start live trading in background via startup script
./startup.sh
```

There are no tests, linters, or build steps configured yet.

## Architecture

**Strategy lifecycle:** `Strategy.compute_indicators(df)` is called once on the full dataset, then `Strategy.on_bar(df, idx, position)` is called for every bar sequentially. When in a position, `Strategy.get_trailing_stop(position, bar)` is called each bar to ratchet the stop. This interface is shared identically between backtest and live trading.

**Strategy registration (`src/strategies/registry.py`):** Uses a `@register('name')` decorator that populates a module-level `_registry` dict. Registration happens at import time — any file that imports a strategy module triggers registration. Strategies are looked up by name at runtime via `get_strategy()`. The optimize script defines additional strategies inline with their own `@register` calls.

**Backtest engine hot loop (`src/backtest/engine.py`):** Per-bar exit checks run in strict priority: liquidation → stop-loss (distinguishes initial vs trailing via the `initial_stop_loss` field) → take-profit → trailing stop update. The `FastBar` class is a numpy-backed proxy that replaces `pd.Series` in the hot loop for ~5x speedup. Cash is modeled as: deduct margin + entry fee on open, release `max(0, margin + pnl - exit_fee)` on close. Liquidation wipes the entire margin (isolated margin model).

**Data models (`src/models.py`):** All dataclasses live in a single file to avoid circular imports — this is deliberate. `Position` tracks both `stop_loss` (current, ratcheted) and `initial_stop_loss` (at entry), which the engine uses to classify exits as `stop_loss` vs `trailing_stop`. `TradeSetup` is what `on_bar()` returns — it only supports entry actions; exits are managed by the engine.

**Configuration (`src/config.py`):** `ConfigLoader.load()` reads `config/default.yaml`, merges runtime `overrides` dict recursively via `_deep_merge`, and returns a `BacktestConfig` dataclass. The `strategy_params` sub-dict is passed directly to the strategy constructor.

**Risk manager (`src/risk/manager.py`):** Computes position size from three modes: `fixed_risk` (size = risk_amount / stop_distance), `fixed_units`, or `percent_equity`. Position size is capped by `capital × leverage / entry_price`. Liquidation price follows Binance's isolated-margin formula: `long: entry × (1 − 1/leverage + mmr)`, `short: entry × (1 + 1/leverage − mmr)`.

**Live trading (`examples/live_trading.py`):** Reimplements the engine's per-bar logic inline in `LiveRunner` rather than importing `BacktestEngine`. Uses ccxt.pro WebSocket to subscribe to Binance kline streams. Warms up indicators with 300 historical bars, then processes each closed candle. Uses the same `FastBar` class from the backtest engine for trailing stop updates.

**Data loading (`src/data/loader.py`):** Fetches OHLCV via ccxt REST API with pagination, caches as CSV to `data/{SYMBOL}_{TIMEFRAME}.csv`. Filters by date range post-fetch. `force_download=True` skips cache.

**Visualization (`src/visualization/plotter.py`):** Four-panel matplotlib dashboard: equity curve (vs buy & hold), drawdown, price chart with entry/exit markers, and per-trade PnL distribution with a metrics summary box. All labels are in Chinese.

## Key design decisions

- **Numpy-backed hot loop (`FastBar`):** The engine pre-converts all DataFrame columns to numpy arrays before the bar loop. `FastBar.__getitem__` reads directly from the numpy array at the current index, eliminating pandas indexing overhead per bar.
- **Single `models.py`:** All dataclasses co-located to prevent circular imports between modules that reference each other's types (engine → position → trade → metrics).
- **Exit logic reimplemented in live runner:** Live trading does not use `BacktestEngine` directly because it runs in an async WebSocket loop and needs different state management. The exit/entry logic is manually replicated.
- **Indicator pre-computation in optimizer:** To avoid recomputing indicators for each parameter combination, the optimizer pre-computes DataFrames keyed by indicator-affecting params, then passes them via `engine.run(strategy, data, indicators_df=precomputed)`.
- **Isolated margin only:** Cross margin is not implemented. Liquidation always wipes the full position margin and cannot exceed it.
