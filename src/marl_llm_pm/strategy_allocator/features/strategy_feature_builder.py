"""Deterministic feature construction from sleeve return streams."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyFeatureSpec:
    """Lookback windows and options controlling feature construction."""

    ret_lookbacks: tuple = (5, 21, 63, 126)    # rolling return windows (periods)
    vol_lookbacks: tuple = (21, 63)             # rolling volatility windows (periods)
    drawdown_lookback: int = 252                # peak-to-trough window (periods)
    corr_lookback: int = 63                     # pairwise correlation window (periods)


def build_strategy_features(
    sleeve_returns: pd.DataFrame,
    *,
    macro_features: Optional[pd.DataFrame] = None,
    spec: StrategyFeatureSpec = StrategyFeatureSpec(),
) -> pd.DataFrame:
    """Build a deterministic feature matrix from sleeve returns.

    Per-sleeve features
    -------------------
    - ``<SLEEVE>__ret_<L>``  : rolling L-period cumulative return
    - ``<SLEEVE>__vol_<L>``  : rolling L-period annualised volatility (std * sqrt(52))
    - ``<SLEEVE>__dd_<W>``   : rolling W-period max drawdown (peak-to-trough)

    Cross-sleeve features
    ---------------------
    - ``mean_pairwise_corr_<L>`` : mean off-diagonal pairwise correlation over L periods

    If ``macro_features`` is supplied its columns are joined on the date index.

    Args:
        sleeve_returns: DataFrame with DatetimeIndex and one column per sleeve.
        macro_features: Optional DataFrame of exogenous macro signals (same index).
        spec: Feature construction parameters.

    Returns:
        DataFrame of features with the same DatetimeIndex (NaN rows from short
        lookbacks are dropped).
    """
    if sleeve_returns.empty:
        raise ValueError("sleeve_returns is empty.")

    sleeves = list(sleeve_returns.columns)
    frames = []

    # Per-sleeve rolling return and volatility
    for sleeve in sleeves:
        s = sleeve_returns[sleeve]

        for L in spec.ret_lookbacks:
            col = f"{sleeve}__ret_{L}"
            frames.append(
                ((1 + s).rolling(L).apply(np.prod, raw=True) - 1).rename(col)
            )

        for L in spec.vol_lookbacks:
            col = f"{sleeve}__vol_{L}"
            frames.append(
                (s.rolling(L).std(ddof=1) * np.sqrt(52)).rename(col)
            )

        W = spec.drawdown_lookback
        col = f"{sleeve}__dd_{W}"
        cumret = (1 + s).rolling(W).apply(
            lambda x: float((np.cumprod(x) / np.maximum.accumulate(np.cumprod(x))).min() - 1),
            raw=True,
        )
        frames.append(cumret.rename(col))

    # Mean pairwise correlation across all sleeves
    L_corr = spec.corr_lookback
    if len(sleeves) > 1:
        def _mean_corr(window_df: pd.DataFrame) -> float:
            c = window_df.corr().values
            n = c.shape[0]
            if n < 2:
                return np.nan
            mask = ~np.eye(n, dtype=bool)
            return float(c[mask].mean())

        mean_corr_series = (
            sleeve_returns.rolling(L_corr)
            .apply(lambda _: np.nan, raw=True)  # placeholder — overwritten below
            [sleeves[0]]
            * np.nan
        )
        # Rolling correlation via explicit loop (pandas rolling corr gives pairwise)
        corr_vals = [np.nan] * len(sleeve_returns)
        arr = sleeve_returns.values
        for i in range(L_corr - 1, len(sleeve_returns)):
            block = arr[i - L_corr + 1 : i + 1]
            df_block = pd.DataFrame(block, columns=sleeves)
            corr_vals[i] = _mean_corr(df_block)

        frames.append(
            pd.Series(corr_vals, index=sleeve_returns.index, name=f"mean_pairwise_corr_{L_corr}")
        )

    features = pd.concat(frames, axis=1)

    if macro_features is not None and not macro_features.empty:
        features = features.join(macro_features, how="left")

    features = features.dropna(how="any")

    if features.empty:
        raise ValueError(
            "Feature matrix is empty after dropping NaNs. "
            "Ensure sleeve_returns has enough history for the requested lookbacks."
        )

    logger.info(f"Built feature matrix: {features.shape} — {list(features.columns)[:6]}...")
    return features


__all__ = ["StrategyFeatureSpec", "build_strategy_features"]
