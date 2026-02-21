"""Test suite for agents and coordination."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from marl_llm_pm.agents import BaseAgent, DummyAgent, AgentCoordinator


class SimpleTestAgent(BaseAgent):
    """Simple test agent for testing."""
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Return fixed weights."""
        return np.ones(self.n_assets) / self.n_assets
    
    def update(self, observation, action, reward, next_observation, done):
        """No-op update."""
        self.step_count += 1


class TestBaseAgent:
    """Test base agent interface."""
    
    def test_agent_creation(self):
        """Test agent initialization."""
        agent = DummyAgent("test_agent", n_assets=3)
        
        assert agent.agent_id == "test_agent"
        assert agent.n_assets == 3
        assert agent.step_count == 0
    
    def test_dummy_agent_action(self):
        """Test dummy agent always returns equal weights."""
        agent = DummyAgent("dummy", n_assets=5)
        observation = np.zeros(10)
        
        action = agent.get_action(observation)
        
        assert action.shape == (5,)
        assert np.allclose(action.sum(), 1.0)
        assert np.allclose(action, 0.2)
    
    def test_agent_reset(self):
        """Test agent reset."""
        agent = DummyAgent("test", n_assets=3)
        agent.step_count = 100
        
        agent.reset()
        
        assert agent.step_count == 0


class TestAgentCoordinator:
    """Test agent coordinator."""
    
    @pytest.fixture
    def agents(self):
        """Create test agents."""
        return [
            DummyAgent(f"agent_{i}", n_assets=3)
            for i in range(3)
        ]
    
    def test_coordinator_creation(self, agents):
        """Test coordinator initialization."""
        coordinator = AgentCoordinator(agents)
        
        assert coordinator.n_agents == 3
        assert coordinator.n_assets == 3
        assert np.allclose(coordinator.agent_weights, 1/3)
    
    def test_coordinator_with_custom_weights(self, agents):
        """Test coordinator with custom agent weights."""
        custom_weights = np.array([0.5, 0.3, 0.2])
        coordinator = AgentCoordinator(agents, weights=custom_weights)
        
        assert np.allclose(coordinator.agent_weights, custom_weights)
    
    def test_aggregation_mean(self, agents):
        """Test mean aggregation."""
        coordinator = AgentCoordinator(agents, aggregation_method="mean")
        
        individual_weights = [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.2, 0.5, 0.3]),
            np.array([0.3, 0.2, 0.5]),
        ]
        
        aggregated = coordinator.aggregate_weights(individual_weights)
        expected = np.array([0.33333333, 0.33333333, 0.33333333])
        
        assert np.allclose(aggregated, expected, atol=0.01)
        assert np.allclose(aggregated.sum(), 1.0)
    
    def test_aggregation_normalization(self, agents):
        """Test that aggregation normalizes weights."""
        coordinator = AgentCoordinator(agents)
        
        # Unnormalized weights
        individual_weights = [
            np.array([1.0, 0.5, 0.0]),
            np.array([0.5, 1.0, 0.5]),
            np.array([0.0, 0.5, 1.0]),
        ]
        
        aggregated = coordinator.aggregate_weights(individual_weights)
        
        assert np.allclose(aggregated.sum(), 1.0)
        assert np.all(aggregated >= 0)
        assert np.all(aggregated <= 1)
    
    def test_get_actions(self, agents):
        """Test getting aggregated action from coordinator."""
        coordinator = AgentCoordinator(agents)
        observation = np.zeros(10)
        
        action = coordinator.get_actions(observation)
        
        assert action.shape == (3,)
        assert np.allclose(action.sum(), 1.0)
    
    def test_update_all_agents(self, agents):
        """Test updating all agents."""
        coordinator = AgentCoordinator(agents)
        
        obs = np.zeros(10)
        action = np.array([1/3, 1/3, 1/3])
        reward = 0.05
        next_obs = np.zeros(10)
        done = False
        
        coordinator.update_all(obs, action, reward, next_obs, done)
        
        for agent in agents:
            assert agent.step_count == 1
    
    def test_reset_all_agents(self, agents):
        """Test resetting all agents."""
        coordinator = AgentCoordinator(agents)
        
        # Advance agents
        for agent in agents:
            agent.step_count = 50
        
        coordinator.reset_all()
        
        for agent in agents:
            assert agent.step_count == 0
    
    def test_diversity_metrics(self, agents):
        """Test ensemble diversity metrics."""
        coordinator = AgentCoordinator(agents)
        metrics = coordinator.get_agent_ensemble_diversity()
        
        assert 'weight_variance' in metrics or 'error' in metrics
        assert metrics.get('num_agents', 0) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
