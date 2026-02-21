"""Multi-agent module for portfolio management."""

from .base_agent import BaseAgent, DummyAgent
from .coordinator import AgentCoordinator

__all__ = ['BaseAgent', 'DummyAgent', 'AgentCoordinator']
