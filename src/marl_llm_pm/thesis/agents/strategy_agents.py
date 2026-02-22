"""Strategy preference agents.

Each agent is responsible for exactly one sleeve.  It observes sleeve-specific
features and emits a preference signal alpha_t in [0, 1].

alpha = 1  → strong preference for this sleeve
alpha = 0  → no preference (effectively zero allocation input)
alpha = 0.5 → neutral

The meta-allocator (orchestrator) converts these signals to final weights.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    x = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class PreferenceOutput:
    """Output of a single strategy agent."""
    sleeve: str
    alpha: float   # in [0, 1]


class StrategyPreferenceAgent:
    """
    Baseline (non-RL) strategy preference agent.

    Reads a scalar 'signal' from obs["sleeve_features"][sleeve_name]["signal"]
    and maps it through a sigmoid to produce alpha in [0, 1].

    This is the correct interface for future RL agents: replace `get_preference`
    with a learned policy that observes the same `obs` dict.

    Expected obs structure
    ----------------------
    obs = {
        "regime_label": str,                          # from RegimeInterpreter
        "sleeve_features": {
            sleeve_name: {
                "signal": float,                      # primary feature
                "momentum": float,                    # optional
                "vol": float,                         # optional
                ...
            }
        },
        "global_features": {                          # market-wide metrics
            "vol": float,
            "drawdown": float,
            "trend": float,
            "corr": float,
        },
        "prev_weights": np.ndarray,                   # last rebalance weights
    }
    """

    def __init__(self, sleeve_name: str, bias: float = 0.0):
        """
        Args:
            sleeve_name: The sleeve this agent is responsible for.
            bias:        Additive bias on the signal before sigmoid (default 0 = neutral).
        """
        self.sleeve_name = sleeve_name
        self.bias = float(bias)

    def get_preference(self, obs: Dict) -> PreferenceOutput:
        feats = obs.get("sleeve_features", {}).get(self.sleeve_name, {})
        signal = float(feats.get("signal", 0.0)) + self.bias
        alpha = _sigmoid(signal)
        return PreferenceOutput(sleeve=self.sleeve_name, alpha=alpha)


def collect_preferences(
    agents: List[StrategyPreferenceAgent],
    obs: Dict,
) -> Dict[str, float]:
    """
    Collect alpha signals from all agents.

    Returns:
        {sleeve_name: alpha} mapping.
    """
    return {a.sleeve_name: a.get_preference(obs).alpha for a in agents}


__all__ = ["StrategyPreferenceAgent", "PreferenceOutput", "collect_preferences"]
