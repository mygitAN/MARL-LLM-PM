import numpy as np
from typing import Dict, List


def safe_simplex(w: np.ndarray) -> np.ndarray:
    """Normalize weights to valid probability simplex."""
    w = np.clip(np.asarray(w, float), 0.0, np.inf)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def apply_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Clip weights to per-sleeve cap and renormalize."""
    w = np.clip(w, 0.0, cap)
    return safe_simplex(w)


class MetaAllocator:
    """Meta-allocator: converts preference signals to portfolio weights.
    
    Enforces mandate-style caps and simplex constraints.
    Can be extended to a learnable policy (RL) later.
    """

    def __init__(self, sleeve_names: List[str], cap: float = 0.70, temperature: float = 1.0):
        self.sleeve_names = sleeve_names
        self.cap = float(cap)
        self.temperature = float(temperature)

    def allocate(self, alphas: Dict[str, float], obs: Dict) -> np.ndarray:
        """Convert preference signals to weights.
        
        Args:
            alphas: Dict mapping sleeve_name → alpha in [0,1]
            obs: Observation dict (optional context)
            
        Returns:
            weights: np.ndarray of length len(sleeve_names), summing to 1
        """
        a = np.array([alphas.get(s, 0.5) for s in self.sleeve_names], dtype=float)
        z = (a - a.max()) / max(self.temperature, 1e-6)
        w = np.exp(z)
        w = safe_simplex(w)
        w = apply_cap(w, self.cap)
        return w
