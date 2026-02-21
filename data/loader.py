"""Market data loader — fetches OHLCV and fundamental data."""

import pandas as pd
from pathlib import Path


class DataLoader:
    """Loads raw market data from local files or external APIs."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """
        Load adjusted close prices for a list of tickers.

        Args:
            tickers: List of ticker symbols.
            start:   Start date (YYYY-MM-DD).
            end:     End date   (YYYY-MM-DD).

        Returns:
            DataFrame with dates as index and tickers as columns.
        """
        raise NotImplementedError("Implement with yfinance / local parquet cache.")

    def load_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """Load fundamental data (P/E, P/B, earnings growth, etc.)."""
        raise NotImplementedError

    def load_news_sentiment(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Load pre-computed news sentiment scores per ticker per day."""
        raise NotImplementedError
