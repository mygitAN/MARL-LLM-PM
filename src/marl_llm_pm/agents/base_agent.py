"""Base agent interface for portfolio management."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional


class BaseAgent(ABC):
    """
    Abstract base class for portfolio management agents.
    
    All agents must implement decision-making logic and track performance.
    """
    
    def __init__(
        self,
        agent_id: str,
        n_assets: int,
        config: Optional[Dict] = None,
    ):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier for this agent
            n_assets: Number of assets in portfolio
            config: Optional configuration dictionary
        """
        self.agent_id = agent_id
        self.n_assets = n_assets
        self.config = config or {}
        self.step_count = 0
        
    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute portfolio weights given market observation.
        
        Args:
            observation: Current market state (from environment)
            
        Returns:
            Portfolio weights for assets
        """
        pass
    
    @abstractmethod
    def update(
        self, 
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Update agent based on experience.
        
        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: New observation
            done: Whether episode finished
        """
        pass
    
    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.step_count = 0
    
    def get_weights(self) -> np.ndarray:
        """Get current portfolio weights (for averaging across agents)."""
        return np.ones(self.n_assets) / self.n_assets
    
    def save(self, path: str) -> None:
        """Save agent state to disk."""
        pass
    
    def load(self, path: str) -> None:
        """Load agent state from disk."""
        pass


class DummyAgent(BaseAgent):
    """Dummy agent that holds equal weights (baseline)."""
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Return equal-weight portfolio."""
        return np.ones(self.n_assets) / self.n_assets
    
    def update(
        self, 
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """No learning for dummy agent."""
        self.step_count += 1


__all__ = ['BaseAgent', 'DummyAgent']
