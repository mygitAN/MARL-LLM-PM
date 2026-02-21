"""Gymnasium-compatible Portfolio Management Environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import pandas as pd


class PortfolioEnv(gym.Env):
    """
    Multi-asset portfolio management environment compatible with Gymnasium.
    
    Agents control portfolio weights across multiple assets.
    Observations include price history, returns, and sentiment signals.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        asset_names: list[str],
        initial_portfolio_value: float = 100000.0,
        max_steps: int = 252,  # 1 trading year
        transaction_cost: float = 0.001,
        seed: Optional[int] = None,
    ):
        """
        Initialize portfolio environment.
        
        Args:
            asset_names: List of asset ticker symbols
            initial_portfolio_value: Starting portfolio value
            max_steps: Maximum trading days per episode
            transaction_cost: Transaction cost as fraction (0.1%)
            seed: Random seed
        """
        super().__init__()
        
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.initial_portfolio_value = initial_portfolio_value
        self.max_steps = max_steps
        self.transaction_cost = transaction_cost
        
        # Environment state
        self.current_step = 0
        self.portfolio_value = initial_portfolio_value
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        self.price_history = None
        self.sentiment_scores = None
        
        # Action space: weights for each asset (must sum to 1)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        
        # Observation space: [price_changes, sentiment_scores, current_weights, portfolio_value_log]
        self.obs_size = self.n_assets * 2 + self.n_assets + 1  # Returns, sentiment, weights, log value
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )
        
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        
    def set_market_data(
        self, 
        price_history: pd.DataFrame,
        sentiment_scores: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Set market price data and optional sentiment scores.
        
        Args:
            price_history: DataFrame with columns as asset names and datetime index
            sentiment_scores: Optional DataFrame with sentiment scores
        """
        self.price_history = price_history
        self.sentiment_scores = sentiment_scores if sentiment_scores is not None else pd.DataFrame()
        self.current_step = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_portfolio_value
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Portfolio weights (must be normalized)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to valid weights
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        # Calculate transaction costs
        weight_changes = np.abs(action - self.portfolio_weights)
        transaction_costs = self.portfolio_value * self.transaction_cost * weight_changes.sum()
        
        # Update portfolio value with price changes
        if self.current_step < len(self.price_history) - 1:
            current_prices = self.price_history.iloc[self.current_step].values
            next_prices = self.price_history.iloc[self.current_step + 1].values
            returns = (next_prices - current_prices) / (current_prices + 1e-8)
            
            # Portfolio return
            portfolio_return = np.dot(self.portfolio_weights, returns)
            self.portfolio_value *= (1 + portfolio_return)
            self.portfolio_value -= transaction_costs
            
            # Reward: portfolio return minus transaction costs
            reward = float(portfolio_return - (transaction_costs / self.portfolio_value))
        else:
            reward = 0.0
        
        # Update weights
        self.portfolio_weights = action
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= len(self.price_history) - 1
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.portfolio_weights.copy(),
            'step': self.current_step,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct current observation."""
        if self.price_history is None or self.current_step >= len(self.price_history):
            # Return zero observation if no data
            return np.zeros(self.obs_size, dtype=np.float32)
        
        obs = []
        
        # Price returns (last window)
        if self.current_step > 0:
            current_prices = self.price_history.iloc[self.current_step].values
            prev_prices = self.price_history.iloc[self.current_step - 1].values
            returns = (current_prices - prev_prices) / (prev_prices + 1e-8)
            obs.extend(returns)
        else:
            obs.extend(np.zeros(self.n_assets))
        
        # Sentiment scores
        if not self.sentiment_scores.empty and self.current_step < len(self.sentiment_scores):
            sentiment = self.sentiment_scores.iloc[self.current_step].values
            obs.extend(sentiment)
        else:
            obs.extend(np.zeros(self.n_assets))
        
        # Current portfolio weights
        obs.extend(self.portfolio_weights)
        
        # Portfolio value (log scale)
        obs.append(np.log(self.portfolio_value + 1))
        
        return np.array(obs, dtype=np.float32)
    
    def render(self) -> None:
        """Render environment state."""
        pass


__all__ = ['PortfolioEnv']
