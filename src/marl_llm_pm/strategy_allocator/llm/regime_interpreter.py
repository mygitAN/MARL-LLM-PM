import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

LABELS = [
    "TRENDING-LOWVOL",
    "STRESS-DRAWDOWN",
    "RECOVERY",
    "SIDEWAYS-HIGHCORR",
    "RISK-OFF-DEFENSIVE",
]


@dataclass
class RegimeOutput:
    """Regime classification output."""
    label: str
    explanation: str


class RegimeInterpreter:
    """Thesis-safe regime interpreter.
    
    - Numeric-only inputs (no free-form text).
    - Closed vocabulary (bounded label set).
    - Deterministic fallback classifier.
    - Optional caching by timestamp key.
    """

    def __init__(self, cache_dir: str = ".cache/regimes", use_cache: bool = True, labels: Optional[List[str]] = None):
        self.labels = labels or LABELS
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key}.json"

    def _load(self, key: str) -> Optional[RegimeOutput]:
        """Load cached regime classification."""
        if not self.use_cache:
            return None
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            if d.get("label") in self.labels:
                return RegimeOutput(label=d["label"], explanation=d.get("explanation", ""))
        except Exception:
            return None
        return None

    def _save(self, key: str, out: RegimeOutput) -> None:
        """Cache regime classification."""
        if not self.use_cache:
            return
        self._cache_path(key).write_text(
            json.dumps({"label": out.label, "explanation": out.explanation}, indent=2)
        )

    def classify(self, key: str, metrics: Dict[str, float]) -> RegimeOutput:
        """Classify regime using deterministic numeric rules.
        
        Expected metrics keys:
            - vol: realized volatility
            - drawdown: rolling drawdown (negative value)
            - trend: rolling return / trend proxy
            - corr: average correlation proxy
        """
        cached = self._load(key)
        if cached:
            return cached

        vol = float(metrics.get("vol", 0.0))
        dd = float(metrics.get("drawdown", 0.0))
        trend = float(metrics.get("trend", 0.0))
        corr = float(metrics.get("corr", 0.0))

        # Deterministic classification rules
        if dd < -0.12 and vol > 0.20:
            label = "STRESS-DRAWDOWN"
            expl = f"High stress: drawdown={dd:.2%}, vol={vol:.2f}."
        elif trend > 0.08 and vol < 0.15:
            label = "TRENDING-LOWVOL"
            expl = f"Trending with low volatility: trend={trend:.2%}, vol={vol:.2f}."
        elif trend > 0.0 and dd > -0.05 and vol < 0.20:
            label = "RECOVERY"
            expl = f"Recovery: trend={trend:.2%}, drawdown={dd:.2%}."
        elif corr > 0.60 and abs(trend) < 0.03:
            label = "SIDEWAYS-HIGHCORR"
            expl = f"Sideways/high correlation: corr={corr:.2f}, trend={trend:.2%}."
        else:
            label = "RISK-OFF-DEFENSIVE"
            expl = f"Defensive/risk-off: vol={vol:.2f}, drawdown={dd:.2%}, trend={trend:.2%}."

        if label not in self.labels:
            label = self.labels[0]
            expl = "Label outside closed set; defaulted."

        out = RegimeOutput(label=label, explanation=expl)
        self._save(key, out)
        return out
