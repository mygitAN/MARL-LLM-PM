"""
Multi-Agent Reinforcement Learning with LLM for Portfolio Management

Thesis pipeline (strategy-sleeve allocator) — primary:
- StrategySleeveEnv       — sleeve return simulator (PnL/costs only)
- StrategyPreferenceAgent — emits alpha_t in [0,1] per sleeve
- MetaAllocator           — converts alphas to w_t with mandate caps
- RegimeInterpreter       — closed-label numeric regime classifier
- proportional_walk_forward — thesis-aligned train/val/test/holdout splits

Legacy pipeline (kept for reference):
- PortfolioEnv, AgentCoordinator, SentimentAnalyzer, Backtester
"""

__version__ = "0.2.0"
__author__ = "MARL-LLM-PM Team"

# ConfigManager has no heavy dependencies — always available
from .config import ConfigManager

# Legacy pipeline — depends on gymnasium; import lazily so data utilities
# can be used in environments where gymnasium is not installed.
try:
    from .environment import PortfolioEnv
    from .agents import BaseAgent, DummyAgent, AgentCoordinator
    from .llm import SentimentAnalyzer
    from .backtesting import BacktestResults, Backtester
except Exception:
    pass

# Thesis pipeline
from .thesis import (
    StrategySleeveEnv,
    StepInfo,
    StrategyPreferenceAgent,
    PreferenceOutput,
    collect_preferences,
    MetaAllocator,
    RegimeInterpreter,
    RegimeOutput,
    LABELS,
    proportional_walk_forward,
    SplitWindow,
)

__all__ = [
    # Legacy
    'PortfolioEnv',
    'BaseAgent',
    'DummyAgent',
    'AgentCoordinator',
    'SentimentAnalyzer',
    'BacktestResults',
    'Backtester',
    'ConfigManager',
    # Thesis
    'StrategySleeveEnv',
    'StepInfo',
    'StrategyPreferenceAgent',
    'PreferenceOutput',
    'collect_preferences',
    'MetaAllocator',
    'RegimeInterpreter',
    'RegimeOutput',
    'LABELS',
    'proportional_walk_forward',
    'SplitWindow',
]
