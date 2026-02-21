"""Return attribution — decompose portfolio returns by agent/factor."""

import pandas as pd
import numpy as np


class ReturnAttribution:
    """
    Brinson-Hood-Beebower style attribution plus per-agent contribution.
    """

    def agent_contribution(
        self,
        agent_weights: dict[str, pd.DataFrame],
        asset_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute each agent's contribution to total portfolio return.

        Args:
            agent_weights: {agent_name: DataFrame(dates x assets)} weight allocations.
            asset_returns: DataFrame(dates x assets) realised returns.

        Returns:
            DataFrame(dates x agents) with daily return contributions.
        """
        contributions = {}
        for name, weights in agent_weights.items():
            aligned_w, aligned_r = weights.align(asset_returns, join="inner")
            contributions[name] = (aligned_w * aligned_r).sum(axis=1)
        return pd.DataFrame(contributions)

    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> pd.Series:
        """
        OLS factor attribution (Fama-French style).

        Args:
            portfolio_returns: Series of daily portfolio returns.
            factor_returns:    DataFrame of factor returns (same index).

        Returns:
            Series of factor loadings (betas).
        """
        from numpy.linalg import lstsq

        y = portfolio_returns.values
        X = np.column_stack([np.ones(len(y)), factor_returns.values])
        betas, _, _, _ = lstsq(X, y, rcond=None)
        labels = ["alpha"] + list(factor_returns.columns)
        return pd.Series(betas, index=labels)
