import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


def sigmoid(x: float) -> float:
    """Sigmoid activation to map signal to [0,1]."""
    return float(1.0 / (1.0 + np.exp(-np.clip(float(x), -100, 100))))


@dataclass
class PreferenceOutput:
    """Strategy agent preference signal output."""
    sleeve: str
    alpha: float  # in [0, 1]


class StrategyPreferenceAgent:
    """Baseline (non-RL) strategy agent.
    
    - Observes sleeve-specific numeric features.
    - Outputs alpha preference signal in [0,1].
    """

    def __init__(self, sleeve_name: str):
        self.sleeve_name = sleeve_name

    def get_preference(self, obs: Dict) -> PreferenceOutput:
        """Compute preference signal from observation.
        
        obs expected to have:
            obs["sleeve_features"][sleeve_name] = dict with "signal" key
        """
        feats = obs.get("sleeve_features", {}).get(self.sleeve_name, {})
        x = feats.get("signal", 0.0)
        return PreferenceOutput(self.sleeve_name, sigmoid(float(x)))


def collect_preferences(agents: List[StrategyPreferenceAgent], obs: Dict) -> Dict[str, float]:
    """Collect preference signals from all agents."""
    return {a.sleeve_name: a.get_preference(obs).alpha for a in agents}
