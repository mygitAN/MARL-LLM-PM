"""Thesis pipeline: strategy-sleeve allocator with regime interpretation."""

from .environment.strategy_env import StrategySleeveEnv, StepInfo
from .agents.strategy_agents import StrategyPreferenceAgent, PreferenceOutput, collect_preferences
from .orchestration.meta_allocator import MetaAllocator
from .llm.regime_interpreter import RegimeInterpreter, RegimeOutput, LABELS
from .evaluation.walk_forward import proportional_walk_forward, SplitWindow

__all__ = [
    "StrategySleeveEnv",
    "StepInfo",
    "StrategyPreferenceAgent",
    "PreferenceOutput",
    "collect_preferences",
    "MetaAllocator",
    "RegimeInterpreter",
    "RegimeOutput",
    "LABELS",
    "proportional_walk_forward",
    "SplitWindow",
]
