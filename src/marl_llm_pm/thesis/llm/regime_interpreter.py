"""Closed-label regime interpreter.

Design constraints (thesis-safe):
  - Numeric-only inputs (no raw text / news)
  - Closed vocabulary (finite label set, defined at init)
  - Deterministic fallback when LLM is disabled or unavailable
  - Output cached by timestamp key to prevent duplicate API calls
  - Temperature = 0 when using LLM (reproducibility)

The optional LLM path calls Claude with a structured prompt that forces
choice within the closed label set.  If the response is outside the set,
the deterministic fallback is used instead.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

LABELS: List[str] = [
    "TRENDING-LOWVOL",
    "STRESS-DRAWDOWN",
    "RECOVERY",
    "SIDEWAYS-HIGHCORR",
    "RISK-OFF-DEFENSIVE",
]


@dataclass
class RegimeOutput:
    """Result of regime classification."""
    label: str        # one of LABELS
    explanation: str  # short human-readable justification


class RegimeInterpreter:
    """
    Thesis-aligned regime interpreter.

    Modes
    -----
    use_llm=False (default):
        Pure deterministic rule-based classifier. No API calls. Fully reproducible.

    use_llm=True:
        Calls Claude with a structured numeric-only prompt.  Response is validated
        against the closed label set; falls back to deterministic rules on failure.

    Cache
    -----
    Results are cached by `key` (typically the date string) so each calendar day
    is classified at most once, regardless of how many pipeline steps share it.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/regimes",
        use_cache: bool = True,
        labels: Optional[List[str]] = None,
        use_llm: bool = False,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        self.labels = list(labels or LABELS)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.use_llm = use_llm
        self.model = model

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        if self.use_llm:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.warning("anthropic package not installed; falling back to deterministic classifier.")
                self.use_llm = False

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"regime_{key}.json"

    def _load(self, key: str) -> Optional[RegimeOutput]:
        if not self.use_cache:
            return None
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            label = d.get("label", "")
            if label in self.labels:
                return RegimeOutput(label=label, explanation=d.get("explanation", ""))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt regime cache entry '{key}': {e}")
        return None

    def _save(self, key: str, out: RegimeOutput) -> None:
        if not self.use_cache:
            return
        try:
            self._cache_path(key).write_text(
                json.dumps({"label": out.label, "explanation": out.explanation}, indent=2)
            )
        except OSError as e:
            logger.warning(f"Could not write regime cache for '{key}': {e}")

    def classify(self, key: str, metrics: Dict[str, float]) -> RegimeOutput:
        """
        Classify the market regime given numeric metrics.

        Args:
            key:     Cache key, typically the date string ("2024-01-15").
            metrics: Numeric market metrics.  Expected keys:
                       vol      – realised volatility (annualised, e.g. 0.18)
                       drawdown – rolling drawdown (negative, e.g. -0.08)
                       trend    – rolling return proxy (e.g. 0.05)
                       corr     – average pairwise sleeve correlation (e.g. 0.65)

        Returns:
            RegimeOutput with a label from the closed set.
        """
        cached = self._load(key)
        if cached:
            return cached

        if self.use_llm and self._client is not None:
            out = self._classify_llm(key, metrics)
        else:
            out = self._classify_rules(metrics)

        self._save(key, out)
        return out

    def _classify_rules(self, metrics: Dict[str, float]) -> RegimeOutput:
        """Deterministic rule-based fallback classifier."""
        vol      = float(metrics.get("vol",      0.0))
        dd       = float(metrics.get("drawdown", 0.0))
        trend    = float(metrics.get("trend",    0.0))
        corr     = float(metrics.get("corr",     0.0))

        if dd < -0.12 and vol > 0.20:
            label = "STRESS-DRAWDOWN"
            expl  = f"High stress: drawdown={dd:.2%}, vol={vol:.2f}."
        elif trend > 0.08 and vol < 0.15:
            label = "TRENDING-LOWVOL"
            expl  = f"Trending/low-vol: trend={trend:.2%}, vol={vol:.2f}."
        elif trend > 0.0 and dd > -0.05 and vol < 0.20:
            label = "RECOVERY"
            expl  = f"Recovery: trend={trend:.2%}, drawdown={dd:.2%}."
        elif corr > 0.60 and abs(trend) < 0.03:
            label = "SIDEWAYS-HIGHCORR"
            expl  = f"Sideways/high-corr: corr={corr:.2f}, trend={trend:.2%}."
        else:
            label = "RISK-OFF-DEFENSIVE"
            expl  = f"Defensive/risk-off: vol={vol:.2f}, dd={dd:.2%}, trend={trend:.2%}."

        # Guard: ensure label is always in closed set
        if label not in self.labels:
            label = self.labels[-1]
            expl += " [defaulted to closed-set fallback]"

        return RegimeOutput(label=label, explanation=expl)

    def _classify_llm(self, key: str, metrics: Dict[str, float]) -> RegimeOutput:
        """
        Optional LLM path — numeric inputs only, closed label set enforced.
        Falls back to deterministic rules if LLM response is invalid.
        """
        labels_str = ", ".join(f'"{l}"' for l in self.labels)
        metrics_str = "\n".join(f"  {k}: {v:.4f}" for k, v in metrics.items())

        prompt = f"""You are a market regime classifier. Classify the current market regime using ONLY the numeric metrics below.

Metrics:
{metrics_str}

You MUST respond with exactly one label from this closed set:
[{labels_str}]

Respond with a single JSON object with exactly two keys:
  "label": one of the labels above (exact string match)
  "explanation": one sentence (max 30 words) citing the numeric evidence

Example:
{{"label": "TRENDING-LOWVOL", "explanation": "Trend=8.2%, vol=12% — sustained low-vol uptrend."}}

Respond with ONLY the JSON object."""

        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=128,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()
            d = json.loads(text)
            label = d.get("label", "")
            expl  = d.get("explanation", "")

            if label not in self.labels:
                logger.warning(
                    f"LLM returned label '{label}' outside closed set for key '{key}'. "
                    "Falling back to deterministic classifier."
                )
                return self._classify_rules(metrics)

            return RegimeOutput(label=label, explanation=expl)

        except Exception as e:
            logger.warning(f"LLM regime classification failed for key '{key}': {e}. Using rule fallback.")
            return self._classify_rules(metrics)


__all__ = ["RegimeInterpreter", "RegimeOutput", "LABELS"]
