import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

EPS = 1e-12


def _safe_simplex(w: np.ndarray) -> np.ndarray:
    """Normalize weights to a valid probability simplex."""
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, np.inf)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def apply_cap_and_renormalize(w: np.ndarray, cap: float) -> np.ndarray:
    """Apply per-sleeve weight cap and renormalize to simplex."""
    w = np.clip(w, 0.0, cap)
    return _safe_simplex(w)


@dataclass
class StepInfo:
    """Information returned from a single environment step."""
    t: int
    portfolio_value: float
    weights: np.ndarray
    turnover: float
    sleeve_return: float


class StrategySleeveEnv:
    """Strategy-sleeve portfolio simulator.
    
    - State is built externally (features + optional regime label); env only simulates PnL/costs.
    - Action is sleeve weights w_t over sleeves (simplex).
    - Reward is 1-step portfolio return net of transaction costs.
    """

    def __init__(
        self,
        sleeve_names: List[str],
        transaction_cost: float = 0.001,
        max_weight_per_sleeve: float = 0.70,
        initial_value: float = 100000.0,
    ):
        self.sleeve_names = list(sleeve_names)
        self.n = len(self.sleeve_names)
        self.tc = float(transaction_cost)
        self.cap = float(max_weight_per_sleeve)
        self.initial_value = float(initial_value)
        self._returns: Optional[pd.DataFrame] = None
        self.reset()

    def set_sleeve_returns(self, sleeve_returns: pd.DataFrame) -> None:
        """Load sleeve returns (columns must match sleeve_names)."""
        missing = set(self.sleeve_names) - set(sleeve_returns.columns)
        if missing:
            raise ValueError(f"Missing sleeve return columns: {sorted(missing)}")
        self._returns = sleeve_returns[self.sleeve_names].copy()

    def reset(self) -> None:
        """Reset environment state."""
        self.t = 0
        self.value = self.initial_value
        self.w = np.ones(self.n) / self.n
        self._peak = self.value

    def step(self, new_w: np.ndarray) -> Tuple[float, StepInfo, bool]:
        """Execute one step with new weights.
        
        Returns:
            reward, info, done
        """
        if self._returns is None:
            raise RuntimeError("Call set_sleeve_returns() before stepping.")
        if self.t >= len(self._returns):
            return 0.0, StepInfo(self.t, self.value, self.w.copy(), 0.0, 0.0), True

        # Enforce constraints and simplex
        new_w = _safe_simplex(new_w)
        new_w = apply_cap_and_renormalize(new_w, self.cap)

        # Turnover + transaction costs
        turnover = float(np.abs(new_w - self.w).sum())
        cost = self.value * self.tc * turnover

        # Sleeve return at time t (previous weights applied to current returns)
        r_vec = self._returns.iloc[self.t].values.astype(float)
        sleeve_return = float(np.dot(self.w, r_vec))

        # Update value
        self.value = self.value * (1.0 + sleeve_return) - cost
        self.value = max(self.value, 1.0)
        self._peak = max(self._peak, self.value)

        # Reward: return minus normalized cost
        reward = sleeve_return - (cost / max(self.value, EPS))

        # Commit weights and increment
        self.w = new_w
        self.t += 1
        done = self.t >= len(self._returns)

        info = StepInfo(
            t=self.t,
            portfolio_value=self.value,
            weights=self.w.copy(),
            turnover=turnover,
            sleeve_return=sleeve_return,
        )
        return float(reward), info, done
