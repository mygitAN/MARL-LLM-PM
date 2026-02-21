"""Test suite for portfolio environment correctness."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from marl_llm_pm.environment import PortfolioEnv


class TestPortfolioEnv:
    """Test PortfolioEnv environment implementation."""
    
    @pytest.fixture
    def env(self):
        """Create a test environment."""
        return PortfolioEnv(
            asset_names=['AAPL', 'GOOGL', 'MSFT'],
            initial_portfolio_value=100000.0,
            max_steps=252,
        )
    
    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.random.uniform(100, 200, size=(100, 3))
        return pd.DataFrame(prices, index=dates, columns=['AAPL', 'GOOGL', 'MSFT'])
    
    def test_env_creation(self, env):
        """Test environment initialization."""
        assert env.n_assets == 3
        assert env.initial_portfolio_value == 100000.0
        assert env.max_steps == 252
        assert env.transaction_cost == 0.001
    
    def test_action_space(self, env):
        """Test action space properties."""
        assert env.action_space.shape == (3,)
        assert np.all(env.action_space.low == 0.0)
        assert np.all(env.action_space.high == 1.0)
    
    def test_observation_space(self, env):
        """Test observation space properties."""
        obs_size = env.n_assets * 2 + env.n_assets + 1
        assert env.observation_space.shape == (obs_size,)
    
    def test_reset(self, env, sample_price_data):
        """Test environment reset."""
        env.set_market_data(sample_price_data)
        obs, info = env.reset()
        
        assert obs.shape == env.observation_space.shape
        assert env.current_step == 0
        assert env.portfolio_value == env.initial_portfolio_value
        assert np.allclose(obs, np.zeros_like(obs), atol=1e-5)
    
    def test_step_basic(self, env, sample_price_data):
        """Test basic step execution."""
        env.set_market_data(sample_price_data)
        env.reset()
        
        # Equal weight action
        action = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert env.current_step == 1
    
    def test_action_normalization(self, env, sample_price_data):
        """Test that actions are properly normalized."""
        env.set_market_data(sample_price_data)
        env.reset()
        
        # Non-normalized action
        action = np.array([1.5, 0.5, 2.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        weights = info['weights']
        assert np.allclose(weights.sum(), 1.0), f"Weights don't sum to 1: {weights}"
        assert np.all(weights >= 0), "Weights should be non-negative"
        assert np.all(weights <= 1), "Weights should not exceed 1"
    
    def test_transaction_costs(self, env, sample_price_data):
        """Test that transaction costs are applied."""
        env.set_market_data(sample_price_data)
        obs, _ = env.reset()
        
        # Initial equal weight
        env.portfolio_weights = np.array([0.33, 0.33, 0.34])
        
        # Drastically change weights
        action = np.array([0.9, 0.05, 0.05], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should have reduced portfolio value due to transaction costs
        assert info['portfolio_value'] < env.initial_portfolio_value
    
    def test_portfolio_value_changes(self, env, sample_price_data):
        """Test portfolio value evolution."""
        env.set_market_data(sample_price_data)
        env.reset()
        
        initial_value = env.portfolio_value
        values = [initial_value]
        
        for _ in range(10):
            action = np.array([1/3, 1/3, 1/3], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            values.append(info['portfolio_value'])
        
        assert len(values) == 11
        # Value should change over time
        assert not all(v == values[0] for v in values)
    
    def test_multiple_steps(self, env, sample_price_data):
        """Test running multiple steps."""
        env.set_market_data(sample_price_data)
        env.reset()
        
        steps_run = 0
        done = False
        
        while not done and steps_run < 50:
            action = np.array([1/3, 1/3, 1/3], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps_run += 1
        
        assert steps_run > 0
        assert steps_run <= 50
    
    def test_sentiment_data_integration(self, env, sample_price_data):
        """Test integration with sentiment scores."""
        sentiment_data = pd.DataFrame(
            np.random.uniform(0, 1, size=(100, 3)),
            index=sample_price_data.index,
            columns=['AAPL', 'GOOGL', 'MSFT']
        )
        
        env.set_market_data(sample_price_data, sentiment_data)
        obs, info = env.reset()
        
        # Should include sentiment data in observation
        assert obs.shape == env.observation_space.shape
    
    def test_deterministic_with_seed(self):
        """Test reproducibility with seed."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.random.uniform(100, 200, size=(100, 3))
        price_data = pd.DataFrame(prices, index=dates, columns=['AAPL', 'GOOGL', 'MSFT'])
        
        # Run 1
        env1 = PortfolioEnv(['AAPL', 'GOOGL', 'MSFT'], seed=42)
        env1.set_market_data(price_data)
        env1.reset()
        
        # Run 2
        env2 = PortfolioEnv(['AAPL', 'GOOGL', 'MSFT'], seed=42)
        env2.set_market_data(price_data)
        env2.reset()
        
        # Should get same initial states
        assert env1.portfolio_value == env2.portfolio_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
