"""MARL-LLM Portfolio Management Framework.

This repo supports two related workflows:

1) A *classic* portfolio RL environment + agents + backtesting.
2) A *thesis-aligned* strategy sleeve allocator (factor rotation).

Some modules depend on optional heavy packages (e.g., `gymnasium`). To keep
lightweight utilities (like CSV builders) usable even when those dependencies
aren't installed yet, we guard most top-level imports.
"""

__version__ = "0.1.0"
__author__ = "Abulele Nxitywa"

__all__: list[str] = [
    "__version__",
    "__author__",
    "ConfigManager",
]

from .config import ConfigManager


# -----------------------------
# Thesis-aligned strategy layer
# -----------------------------
try:
    from .strategy_allocator import (
        StrategySleeveEnv,
        StrategySelectorAgent,
        MetaAllocator,
        RegimeInterpreter,
    )

    __all__ += [
        "StrategySleeveEnv",
        "StrategySelectorAgent",
        "MetaAllocator",
        "RegimeInterpreter",
    ]
except Exception:
    # If optional deps for the strategy layer are missing, keep base utilities importable.
    pass


# -----------------------------
# Legacy multi-asset RL layer
# -----------------------------
try:
    from .environment import PortfolioEnv
    from .agents import BaseAgent, DummyAgent, AgentCoordinator
    from .llm import SentimentAnalyzer
    from .backtesting import Backtester, BacktestResults

    __all__ += [
        "PortfolioEnv",
        "BaseAgent",
        "DummyAgent",
        "AgentCoordinator",
        "SentimentAnalyzer",
        "Backtester",
        "BacktestResults",
    ]
except Exception:
    # Most commonly: gymnasium not installed yet.
    pass
