"""Meta-allocator / orchestrator.

Converts per-sleeve preference signals (alpha_t) into final portfolio weights (w_t)
subject to mandate-style constraints (per-sleeve cap, simplex).

This is the component you will eventually replace with a learnable policy (RL agent).
The baseline implementation uses temperature-scaled softmax of the alpha signals.
"""

import numpy as np
from typing import Dict, List


def _safe_simplex(w: np.ndarray) -> np.ndarray:
    """Project onto the probability simplex."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, np.inf)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def _apply_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Clip each weight to `cap`, then renormalise."""
    w = np.clip(w, 0.0, cap)
    return _safe_simplex(w)


class MetaAllocator:
    """
    Baseline meta-allocator: preference signals → constrained portfolio weights.

    Algorithm
    ---------
    1. Stack alpha signals into a vector.
    2. Apply temperature-scaled softmax (stable, bounded output).
    3. Enforce per-sleeve cap + renormalise.

    The temperature parameter controls how "peaked" the allocation is:
      temperature → 0  : winner-takes-all (up to cap)
      temperature = 1  : standard softmax of alphas
      temperature → ∞  : equal-weight regardless of alphas

    Notes
    -----
    - The regime label is available in `obs` for future conditional logic
      (e.g. tighten caps in STRESS-DRAWDOWN regimes).
    - Replace `allocate()` with a learned policy to implement RL meta-allocator.
    """

    def __init__(
        self,
        sleeve_names: List[str],
        cap: float = 0.70,
        temperature: float = 1.0,
    ):
        """
        Args:
            sleeve_names: Ordered list of sleeve identifiers (must match env).
            cap:          Hard per-sleeve weight cap (mandate constraint).
            temperature:  Softmax temperature for converting alphas to weights.
        """
        self.sleeve_names = list(sleeve_names)
        self.cap = float(cap)
        self.temperature = max(float(temperature), 1e-6)

    def allocate(self, alphas: Dict[str, float], obs: Dict) -> np.ndarray:
        """
        Convert preference signals to constrained portfolio weights.

        Args:
            alphas: {sleeve_name: alpha} mapping from strategy agents.
            obs:    Full observation dict (available for future conditional logic).

        Returns:
            np.ndarray of shape (n_sleeves,), sums to 1.0, each element ≤ cap.
        """
        a = np.array(
            [alphas.get(s, 0.5) for s in self.sleeve_names],
            dtype=float,
        )
        # Numerically stable softmax with temperature
        z = (a - a.max()) / self.temperature
        w = np.exp(z)
        w = _safe_simplex(w)
        w = _apply_cap(w, self.cap)
        return w

    def verify_constraints(self, w: np.ndarray) -> bool:
        """
        Assert that output weights satisfy mandate constraints.

        Returns:
            True if all constraints are satisfied.
        """
        tol = 1e-6
        if not np.isclose(w.sum(), 1.0, atol=tol):
            return False
        if np.any(w < -tol) or np.any(w > self.cap + tol):
            return False
        return True


__all__ = ["MetaAllocator"]
