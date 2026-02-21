"""Feature engineering and data preprocessing pipeline."""

import pandas as pd
import numpy as np


class DataPreprocessor:
    """Transforms raw OHLCV data into model-ready features."""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns from price series."""
        return np.log(prices / prices.shift(1)).dropna()

    def compute_momentum_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Rolling momentum signals: 1m, 3m, 6m, 12m returns."""
        windows = [21, 63, 126, 252]
        features = {}
        for w in windows:
            features[f"mom_{w}d"] = prices.pct_change(w)
        return pd.DataFrame(features)

    def compute_volatility(self, returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """Rolling realised volatility."""
        return returns.rolling(window).std() * np.sqrt(252)

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score normalisation."""
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    def build_feature_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline: prices -> normalised feature matrix ready for agents.
        Returns a MultiIndex DataFrame (date, ticker).
        """
        returns = self.compute_returns(prices)
        mom = self.compute_momentum_features(prices)
        vol = self.compute_volatility(returns)
        # Combine and forward-fill gaps
        combined = pd.concat([returns, mom, vol], axis=1).ffill().dropna()
        return self.normalise(combined)
