from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyFeatureSpec:
    """Deterministic, auditable features for strategy rotation."""

    # Lookbacks are in trading days (assumes daily input)
    ret_lookbacks: tuple[int, ...] = (5, 21, 63, 126)
    vol_lookbacks: tuple[int, ...] = (21, 63)
    drawdown_lookback: int = 252
    corr_lookback: int = 63


def _rolling_drawdown(cum: pd.Series, lookback: int) -> pd.Series:
    roll_max = cum.rolling(lookback, min_periods=max(10, lookback // 10)).max()
    return cum / roll_max - 1.0


def build_strategy_features(
    sleeve_returns: pd.DataFrame,
    *,
    macro_features: Optional[pd.DataFrame] = None,
    spec: StrategyFeatureSpec = StrategyFeatureSpec(),
) -> pd.DataFrame:
    """Build a dense feature matrix aligned with sleeve returns."""

    if sleeve_returns.empty:
        raise ValueError("sleeve_returns is empty")
    if not isinstance(sleeve_returns.index, pd.DatetimeIndex):
        raise ValueError("sleeve_returns must be indexed by dates")

    sr = sleeve_returns.sort_index().copy()
    feats: list[pd.DataFrame] = []

    for L in spec.ret_lookbacks:
        r = sr.rolling(L, min_periods=max(3, L // 5)).sum()
        r.columns = [f"{c}__ret_{L}" for c in r.columns]
        feats.append(r)

    for L in spec.vol_lookbacks:
        v = sr.rolling(L, min_periods=max(5, L // 5)).std()
        v.columns = [f"{c}__vol_{L}" for c in v.columns]
        feats.append(v)

    cum = (1.0 + sr).cumprod()
    dd = pd.DataFrame(index=sr.index)
    for c in sr.columns:
        dd[f"{c}__dd_{spec.drawdown_lookback}"] = _rolling_drawdown(
            cum[c], spec.drawdown_lookback
        )
    feats.append(dd)

    if sr.shape[1] >= 2:
        def mean_pairwise_corr(window_df: pd.DataFrame) -> float:
            x = window_df.dropna(how="any").values
            if x.shape[0] < 5:
                return np.nan
            corr = np.corrcoef(x, rowvar=False)
            iu = np.triu_indices_from(corr, k=1)
            return float(np.nanmean(corr[iu]))

        # Rolling apply operates column-wise; we compute once using the full window
        # by applying to a single column but passing the entire window via closure.
        # Easiest: compute directly with rolling over sr using a custom loop.
        mpc = []
        idx = sr.index
        L = spec.corr_lookback
        min_p = max(10, L // 5)
        for i in range(len(sr)):
            if i + 1 < min_p:
                mpc.append(np.nan)
                continue
            start = max(0, i + 1 - L)
            mpc.append(mean_pairwise_corr(sr.iloc[start : i + 1]))
        mpc_series = pd.Series(mpc, index=idx, name=f"SLEEVE__mean_pairwise_corr_{L}")
        feats.append(mpc_series.to_frame())

    feature_df = pd.concat(feats, axis=1)

    if macro_features is not None:
        feature_df = feature_df.join(macro_features.sort_index(), how="left")

    feature_df = feature_df.dropna(axis=0, how="any")
    if feature_df.empty:
        raise ValueError(
            "Feature matrix is empty after dropping NaNs. "
            "Check history length and lookbacks."
        )

    return feature_df
