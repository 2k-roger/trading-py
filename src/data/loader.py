"""Fetch and cache OHLCV data from ccxt exchanges."""

from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent.parent / 'data'


class DataLoader:
    """Fetch OHLCV data from a ccxt exchange with local CSV caching."""

    def __init__(self, exchange_id: str = 'binance'):
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({'enableRateLimit': True})
        if not self.exchange.has.get('fetchOHLCV'):
            raise ValueError(f"{exchange_id} does not support fetchOHLCV")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        symbol: str,
        timeframe: str = '1h',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV data, using cache if available.

        Args:
            symbol: Trading pair, e.g. 'ETH/USDT'.
            timeframe: Bar resolution, e.g. '1h', '15m', '1d'.
            start_date: ISO format, e.g. '2023-01-01'. None = earliest available.
            end_date: ISO format for filtering. None = latest available.
            force_download: Skip cache and re-fetch.

        Returns:
            DataFrame with columns: open, high, low, close, volume, indexed by timestamp.
        """
        cache_path = self._cache_path(symbol, timeframe)

        if not force_download and cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index.name = 'timestamp'
        else:
            since_ms = self._parse_start_ms(start_date)
            bars = self._fetch_all(symbol, timeframe, since_ms)
            if not bars:
                raise ValueError(
                    f"No data returned for {symbol} {timeframe} from {start_date or 'earliest'}"
                )
            df = self._to_dataframe(bars)
            df.to_csv(cache_path)

        # Filter by date range
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_all(
        self, symbol: str, timeframe: str, since_ms: int, limit: int = 1000
    ) -> list:
        """Paginate through all available OHLCV bars."""
        all_bars = []
        while True:
            bars = self.exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=limit
            )
            if not bars:
                break
            all_bars.extend(bars)
            if len(bars) < limit:
                break
            since_ms = bars[-1][0] + 1  # next candle timestamp
        return all_bars

    @staticmethod
    def _to_dataframe(bars: list) -> pd.DataFrame:
        """Convert ccxt raw response to clean DataFrame."""
        df = pd.DataFrame(
            bars,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        df.index.name = 'timestamp'
        # Drop any duplicate index entries
        df = df[~df.index.duplicated(keep='first')]
        return df

    @staticmethod
    def _cache_path(symbol: str, timeframe: str) -> Path:
        """Build a filename-safe cache path."""
        safe = symbol.replace('/', '_').replace(':', '_')
        return CACHE_DIR / f"{safe}_{timeframe}.csv"

    @staticmethod
    def _parse_start_ms(start_date: Optional[str]) -> int:
        """Convert ISO date to milliseconds timestamp."""
        if start_date is None:
            return 0
        ts = pd.Timestamp(start_date)
        return int(ts.timestamp() * 1000)
