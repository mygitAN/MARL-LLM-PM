"""Agent coordinator for multi-agent portfolio weight aggregation."""

import logging
import numpy as np
from typing import List, Dict, Optional
from .base_agent import BaseAgent
from ..constants import EPSILON

logger = logging.getLogger(__name__)


def safe_normalize(weights: np.ndarray) -> np.ndarray:
    """Normalise weights to sum to 1; falls back to equal weights on degenerate input."""
    total = weights.sum()
    if total == 0 or not np.isfinite(total):
        return np.ones_like(weights) / len(weights)
    return weights / total


class AgentCoordinator:
    """
    Coordinates multiple agents and aggregates their portfolio weights.
    
    Implements weight averaging strategies and ensures valid portfolio allocation.
    """
    
    def __init__(
        self,
        agents: List[BaseAgent],
        aggregation_method: str = "mean",
        weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize coordinator.
        
        Args:
            agents: List of BaseAgent instances
            aggregation_method: How to combine weights ("mean", "median", "weighted")
            weights: Optional agent weight multipliers (must sum to 1)
        """
        self.agents = agents
        self.n_agents = len(agents)
        self.aggregation_method = aggregation_method
        
        # Validate and set agent weights
        if weights is not None:
            assert len(weights) == self.n_agents, "Weights length must match agent count"
            assert np.isclose(weights.sum(), 1.0), "Weights must sum to 1"
            self.agent_weights = np.array(weights, dtype=np.float32)
        else:
            self.agent_weights = np.ones(self.n_agents) / self.n_agents
        
        self.n_assets = agents[0].n_assets if agents else 0
        
    def aggregate_weights(self, individual_weights: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate weights from multiple agents.
        
        Args:
            individual_weights: List of weight arrays from each agent
            
        Returns:
            Aggregated portfolio weights (normalized)
        """
        if not individual_weights:
            return np.ones(self.n_assets) / self.n_assets
        
        weights_array = np.array(individual_weights, dtype=np.float32)
        
        if self.aggregation_method == "mean":
            aggregated = np.average(weights_array, axis=0, weights=self.agent_weights)
        elif self.aggregation_method == "median":
            aggregated = np.median(weights_array, axis=0)
        elif self.aggregation_method == "weighted":
            aggregated = np.average(weights_array, axis=0, weights=self.agent_weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        # Normalize and clip to ensure valid weights
        aggregated = np.clip(aggregated, 0, 1)
        aggregated = safe_normalize(aggregated)
        
        return aggregated
    
    def get_actions(self, observation: np.ndarray) -> np.ndarray:
        """
        Get aggregated action from all agents.
        
        Args:
            observation: Current market observation
            
        Returns:
            Aggregated portfolio weights
        """
        individual_actions = [
            agent.get_action(observation) for agent in self.agents
        ]
        aggregated_action = self.aggregate_weights(individual_actions)
        return aggregated_action
    
    def update_all(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Update all agents with shared experience.
        
        Args:
            observation: Previous observation
            action: Action taken (aggregated)
            reward: Reward received
            next_observation: New observation
            done: Whether episode finished
        """
        for agent in self.agents:
            agent.update(observation, action, reward, next_observation, done)
    
    def reset_all(self) -> None:
        """Reset all agents."""
        for agent in self.agents:
            agent.reset()
    
    def get_agent_ensemble_diversity(self) -> Dict[str, float]:
        """
        Assess diversity of agent outputs for current observation.
        
        Returns:
            Dictionary with diversity metrics
        """
        if not self.agents:
            return {}
        
        dummy_obs = np.zeros((self.n_agents * 2 + self.n_assets + 1,), dtype=np.float32)
        
        try:
            individual_weights = [
                agent.get_action(dummy_obs) for agent in self.agents
            ]
            weights_array = np.array(individual_weights)
            
            # Calculate variance across agents per asset
            variance = np.var(weights_array, axis=0).mean()
            
            # Calculate pairwise differences
            max_pairwise_diff = 0.0
            for i in range(len(individual_weights)):
                for j in range(i + 1, len(individual_weights)):
                    diff = np.abs(individual_weights[i] - individual_weights[j]).sum()
                    max_pairwise_diff = max(max_pairwise_diff, diff)
            
            return {
                "weight_variance": float(variance),
                "max_pairwise_difference": float(max_pairwise_diff),
                "num_agents": self.n_agents,
            }
        except Exception as e:
            logger.warning(f"Could not compute diversity metrics: {e}", exc_info=True)
            return {"error": f"Could not compute diversity metrics: {type(e).__name__}"}
    
    def save_all(self, checkpoint_dir: str) -> None:
        """Save all agents to checkpoint directory."""
        for i, agent in enumerate(self.agents):
            agent.save(f"{checkpoint_dir}/agent_{i}")
    
    def load_all(self, checkpoint_dir: str) -> None:
        """Load all agents from checkpoint directory."""
        for i, agent in enumerate(self.agents):
            agent.load(f"{checkpoint_dir}/agent_{i}")


__all__ = ['AgentCoordinator']
