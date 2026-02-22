"""Strategy-sleeve portfolio simulator.

The environment is intentionally minimal: it owns PnL / cost accounting only.
Feature engineering and observation construction live outside (in main pipeline),
so the env stays a clean, reusable simulation primitive.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

EPS = 1e-12


def _safe_simplex(w: np.ndarray) -> np.ndarray:
    """Project weights onto the probability simplex (non-negative, sum-to-one)."""
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, np.inf)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def apply_cap_and_renormalize(w: np.ndarray, cap: float) -> np.ndarray:
    """
    Clip each sleeve weight to `cap`, then renormalise to the simplex.

    This is a deterministic per-element cap (not a full convex projection),
    which is stable and adequate for thesis mandate-style constraints.
    """
    w = np.clip(w, 0.0, cap)
    return _safe_simplex(w)


@dataclass
class StepInfo:
    """Snapshot of environment state after one step."""
    t: int
    portfolio_value: float
    weights: np.ndarray
    turnover: float
    sleeve_return: float


class StrategySleeveEnv:
    """
    Strategy-sleeve portfolio simulator.

    Sleeves are pre-computed return series (e.g. factor indices or backtest streams).
    The env does not know how weights are computed — it only simulates PnL and costs.

    Observation construction (regime label, numeric features, prev weights) is the
    responsibility of the outer pipeline loop.

    Action space: sleeve weight vector on the simplex, subject to per-sleeve cap.
    Reward:       1-step net-of-costs portfolio return.
    """

    def __init__(
        self,
        sleeve_names: List[str],
        transaction_cost: float = 0.001,
        max_weight_per_sleeve: float = 0.70,
        initial_value: float = 100_000.0,
    ):
        """
        Args:
            sleeve_names:           Ordered list of sleeve identifiers.
            transaction_cost:       One-way cost applied to weight turnover (fraction).
            max_weight_per_sleeve:  Hard cap per sleeve (mandate constraint).
            initial_value:          Starting portfolio NAV.
        """
        self.sleeve_names = list(sleeve_names)
        self.n = len(self.sleeve_names)
        self.tc = float(transaction_cost)
        self.cap = float(max_weight_per_sleeve)
        self.initial_value = float(initial_value)

        self._returns: Optional[pd.DataFrame] = None
        self.reset()

    def set_sleeve_returns(self, sleeve_returns: pd.DataFrame) -> None:
        """
        Attach sleeve return data.  Must be called before any episode.

        Args:
            sleeve_returns: DataFrame with DatetimeIndex, columns = sleeve_names.
                            Values are period returns (not log-returns, not prices).
        """
        missing = set(self.sleeve_names) - set(sleeve_returns.columns)
        if missing:
            raise ValueError(f"Missing sleeve return columns: {sorted(missing)}")
        self._returns = sleeve_returns[self.sleeve_names].copy()

    def reset(self) -> None:
        """Reset to start of episode."""
        self.t = 0
        self.value = self.initial_value
        self.w = np.ones(self.n) / self.n
        self._peak = self.value

    def step(self, new_w: np.ndarray) -> Tuple[float, StepInfo, bool]:
        """
        Execute one rebalance step.

        Args:
            new_w: Proposed sleeve weights (will be projected to simplex + cap).

        Returns:
            (reward, StepInfo, done)
            reward: 1-step net return minus proportional transaction cost.
            done:   True when the return series is exhausted.
        """
        if self._returns is None:
            raise RuntimeError("Call set_sleeve_returns() before stepping.")

        if self.t >= len(self._returns):
            return 0.0, StepInfo(self.t, self.value, self.w.copy(), 0.0, 0.0), True

        new_w = _safe_simplex(new_w)
        new_w = apply_cap_and_renormalize(new_w, self.cap)

        turnover = float(np.abs(new_w - self.w).sum())
        cost = self.value * self.tc * turnover

        # Realised sleeve return at step t using *previous* weights (pre-rebalance)
        r_vec = self._returns.iloc[self.t].values.astype(float)
        sleeve_return = float(np.dot(self.w, r_vec))

        self.value = self.value * (1.0 + sleeve_return) - cost
        self.value = max(self.value, 1.0)
        self._peak = max(self._peak, self.value)

        reward = sleeve_return - (cost / max(self.value, EPS))

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

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak NAV."""
        return (self.value - self._peak) / max(self._peak, EPS)

    @property
    def n_steps(self) -> int:
        """Total number of steps in the loaded return series."""
        return len(self._returns) if self._returns is not None else 0


__all__ = ["StrategySleeveEnv", "StepInfo"]
