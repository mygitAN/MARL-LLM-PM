"""Backtesting framework with performance metrics."""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..constants import EPSILON

logger = logging.getLogger(__name__)


def safe_normalize(weights: np.ndarray) -> np.ndarray:
    """Normalise weights to sum to 1; falls back to equal weights on degenerate input."""
    total = weights.sum()
    if total == 0 or not np.isfinite(total):
        return np.ones_like(weights) / len(weights)
    return weights / total


@dataclass
class BacktestResults:
    """
    Container for backtest performance metrics.
    
    Tracks returns, drawdown, Sharpe ratio, and other key statistics.
    """
    
    portfolio_values: np.ndarray
    returns: np.ndarray
    dates: Optional[pd.DatetimeIndex] = None
    benchmark_returns: Optional[np.ndarray] = None
    
    # Computed metrics
    total_return: float = field(default=0.0)
    annualized_return: float = field(default=0.0)
    annualized_volatility: float = field(default=0.0)
    sharpe_ratio: float = field(default=0.0)
    max_drawdown: float = field(default=0.0)
    cumulative_max_drawdown: float = field(default=0.0)
    sortino_ratio: float = field(default=0.0)
    win_rate: float = field(default=0.0)
    alpha: float = field(default=0.0)
    beta: float = field(default=0.0)
    calmar_ratio: float = field(default=0.0)
    
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute all metrics upon initialization."""
        self._compute_metrics()
    
    def _compute_metrics(self) -> None:
        """Compute all performance metrics."""
        if len(self.returns) == 0:
            return
        
        self.total_return = float((self.portfolio_values[-1] / self.portfolio_values[0]) - 1)

        # Annualization (assume 252 trading days)
        trading_days = len(self.returns)
        years = trading_days / 252.0

        if years > 0:
            self.annualized_return = float(
                (1 + self.total_return) ** (1 / years) - 1
            )

        self.annualized_volatility = float(
            np.std(self.returns, ddof=1) * np.sqrt(252)
        )

        # Sharpe Ratio (assuming 0% risk-free rate)
        if self.annualized_volatility > 0:
            self.sharpe_ratio = float(
                self.annualized_return / self.annualized_volatility
            )

        cumulative = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        self.max_drawdown = float(drawdowns.min())
        self.cumulative_max_drawdown = float(drawdowns.min())

        if abs(self.max_drawdown) > 1e-6:
            self.calmar_ratio = float(
                self.annualized_return / abs(self.max_drawdown)
            )

        # Sortino Ratio (downside volatility)
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) > 0:
            downside_vol = float(
                np.std(downside_returns, ddof=1) * np.sqrt(252)
            )
            if downside_vol > 0:
                self.sortino_ratio = float(
                    self.annualized_return / downside_vol
                )

        winning_days = np.sum(self.returns > 0)
        self.win_rate = float(winning_days / len(self.returns))

        if self.benchmark_returns is not None and len(self.benchmark_returns) == len(self.returns):
            covariance = np.cov(self.returns, self.benchmark_returns)[0, 1]
            benchmark_var = np.var(self.benchmark_returns, ddof=1)
            
            if benchmark_var > 0:
                self.beta = float(covariance / benchmark_var)
                benchmark_mean_ret = np.mean(self.benchmark_returns)
                self.alpha = float(
                    np.mean(self.returns) - (self.beta * benchmark_mean_ret)
                )
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'annualized_volatility': self.annualized_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'sortino_ratio': self.sortino_ratio,
            'win_rate': self.win_rate,
            'alpha': self.alpha,
            'beta': self.beta,
            'calmar_ratio': self.calmar_ratio,
        }
    
    def summary(self) -> str:
        """Print summary of results."""
        lines = [
            "=" * 50,
            "BACKTEST RESULTS SUMMARY",
            "=" * 50,
            f"Total Return:          {self.total_return:>10.2%}",
            f"Annualized Return:     {self.annualized_return:>10.2%}",
            f"Annualized Volatility: {self.annualized_volatility:>10.2%}",
            f"Sharpe Ratio:          {self.sharpe_ratio:>10.2f}",
            f"Sortino Ratio:         {self.sortino_ratio:>10.2f}",
            f"Max Drawdown:          {self.max_drawdown:>10.2%}",
            f"Calmar Ratio:          {self.calmar_ratio:>10.2f}",
            f"Win Rate:              {self.win_rate:>10.2%}",
            "=" * 50,
        ]
        return "\n".join(lines)


class Backtester:
    """
    Backtesting engine for portfolio strategies.
    
    Executes a trading strategy on historical data and computes metrics.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        rebalance_frequency: str = "daily",
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting portfolio value
            transaction_cost: Transaction cost as fraction (0.1%)
            rebalance_frequency: How often to rebalance ("daily", "weekly", "monthly")
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency
    
    def run(
        self,
        price_history: pd.DataFrame,
        weight_calculator: callable,
        benchmark_returns: Optional[np.ndarray] = None,
        **kwargs
    ) -> BacktestResults:
        """
        Run backtest with weight calculation function.
        
        Args:
            price_history: DataFrame with asset prices (datetime index)
            weight_calculator: Function that takes observation and returns weights
            benchmark_returns: Optional benchmark returns for alpha/beta calculation
            **kwargs: Additional arguments for weight_calculator
            
        Returns:
            BacktestResults object with performance metrics
        """
        if len(price_history) < 2:
            raise ValueError("Price history must have at least 2 data points")

        portfolio_values = [self.initial_capital]
        portfolio_weights = np.ones(len(price_history.columns)) / len(price_history.columns)
        returns_list = []

        for i in range(1, len(price_history)):
            current_prices = price_history.iloc[i-1].values
            next_prices = price_history.iloc[i].values

            price_returns = (next_prices - current_prices) / np.where(
                current_prices != 0, current_prices, EPSILON
            )

            try:
                observation = self._build_observation(price_history, i, portfolio_weights)
                new_weights = weight_calculator(observation, **kwargs)
                new_weights = safe_normalize(np.clip(new_weights, 0, 1))
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Weight calculation failed at step {i}: {e}. Holding previous weights.")
                new_weights = portfolio_weights.copy()
            except Exception as e:
                logger.error(f"Unexpected error in weight calculator at step {i}: {e}", exc_info=True)
                new_weights = portfolio_weights.copy()

            weight_diff = np.abs(new_weights - portfolio_weights)
            transaction_cost_amt = portfolio_values[-1] * self.transaction_cost * weight_diff.sum()

            portfolio_return = np.dot(portfolio_weights, price_returns)
            new_value = portfolio_values[-1] * (1 + portfolio_return) - transaction_cost_amt

            portfolio_values.append(max(new_value, 1))  # floor NAV at 1
            returns_list.append(portfolio_return)
            portfolio_weights = new_weights

        returns_array = np.array(returns_list)
        portfolio_values_array = np.array(portfolio_values)
        
        results = BacktestResults(
            portfolio_values=portfolio_values_array,
            returns=returns_array,
            dates=price_history.index,
            benchmark_returns=benchmark_returns,
            metadata={
                'initial_capital': self.initial_capital,
                'transaction_cost': self.transaction_cost,
                'num_assets': len(price_history.columns),
                'backtest_period': f"{price_history.index[0]} to {price_history.index[-1]}",
            }
        )
        
        return results
    
    def _build_observation(
        self,
        price_history: pd.DataFrame,
        step: int,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """Build observation vector from price history."""
        n_assets = len(price_history.columns)
        
        if step > 0:
            current_prices = price_history.iloc[step].values
            prev_prices = price_history.iloc[step - 1].values
            returns = (current_prices - prev_prices) / np.where(prev_prices != 0, prev_prices, EPSILON)
        else:
            returns = np.zeros(n_assets)
        
        observation = np.concatenate([returns, current_weights, [np.log(step + 1)]])
        return observation.astype(np.float32)
    
    @staticmethod
    def calculate_metrics(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, float]:
        """Calculate performance metrics from returns."""
        metrics = {}

        trading_days = len(returns)
        years = trading_days / 252.0

        total_return = np.prod(1 + returns) - 1
        metrics['total_return'] = float(total_return)

        if years > 0:
            metrics['annualized_return'] = float((1 + total_return) ** (1 / years) - 1)

        volatility = np.std(returns, ddof=1)
        metrics['annualized_volatility'] = float(volatility * np.sqrt(252))

        if metrics['annualized_volatility'] > 0:
            metrics['sharpe_ratio'] = float(
                (metrics['annualized_return'] - risk_free_rate) / metrics['annualized_volatility']
            )

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = float(drawdowns.min())
        
        return metrics


__all__ = ['BacktestResults', 'Backtester']
