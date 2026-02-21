"""Performance metrics for portfolio evaluation."""

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """Compute standard portfolio performance statistics."""

    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate

    def total_return(self, returns: pd.Series) -> float:
        return (1 + returns).prod() - 1

    def annualised_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        n = len(returns)
        return (1 + self.total_return(returns)) ** (periods_per_year / n) - 1

    def annualised_volatility(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        return returns.std() * np.sqrt(periods_per_year)

    def sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        excess = returns - self.risk_free_rate / periods_per_year
        if excess.std() == 0:
            return 0.0
        return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)

    def sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        excess = returns - self.risk_free_rate / periods_per_year
        downside = excess[excess < 0].std()
        if downside == 0:
            return 0.0
        return (excess.mean() / downside) * np.sqrt(periods_per_year)

    def max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def calmar_ratio(self, returns: pd.Series) -> float:
        ann_ret = self.annualised_return(returns)
        mdd = abs(self.max_drawdown(returns))
        return ann_ret / mdd if mdd != 0 else 0.0

    def summary(self, returns: pd.Series) -> dict:
        return {
            "total_return":        self.total_return(returns),
            "annualised_return":   self.annualised_return(returns),
            "annualised_vol":      self.annualised_volatility(returns),
            "sharpe_ratio":        self.sharpe_ratio(returns),
            "sortino_ratio":       self.sortino_ratio(returns),
            "max_drawdown":        self.max_drawdown(returns),
            "calmar_ratio":        self.calmar_ratio(returns),
        }
