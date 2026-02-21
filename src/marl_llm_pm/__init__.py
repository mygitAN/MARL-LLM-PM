"""
Multi-Agent Reinforcement Learning with LLM for Portfolio Management

A framework for building intelligent portfolio management systems using:
- Gymnasium-compatible environment for portfolio operations
- Multi-agent coordination with weight aggregation
- Claude LLM-powered sentiment analysis with daily caching
- Backtesting engine with Sharpe/drawdown metrics
"""

__version__ = "0.1.0"
__author__ = "MARL-LLM-PM Team"

from .environment import PortfolioEnv
from .agents import BaseAgent, DummyAgent, AgentCoordinator
from .llm import SentimentAnalyzer
from .backtesting import BacktestResults, Backtester
from .config import ConfigManager

__all__ = [
    'PortfolioEnv',
    'BaseAgent',
    'DummyAgent',
    'AgentCoordinator',
    'SentimentAnalyzer',
    'BacktestResults',
    'Backtester',
    'ConfigManager',
]
