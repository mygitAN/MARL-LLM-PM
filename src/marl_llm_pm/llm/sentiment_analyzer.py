"""Claude-powered sentiment analysis with daily caching."""

import json
import hashlib
import logging
import os
import stat
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import anthropic

from ..constants import DEFAULT_LLM_MODEL, MAX_CACHE_SIZE_MB, API_RATE_LIMIT_PER_MINUTE

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Token-bucket rate limiter for API calls."""

    def __init__(self, calls_per_minute: int = API_RATE_LIMIT_PER_MINUTE):
        self._min_interval = 60.0 / calls_per_minute
        self._last_call: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


class SentimentAnalyzer:
    """
    Analyzes market sentiment for portfolio assets using Claude LLM.

    Features:
    - Daily caching to avoid redundant API calls
    - Multi-asset batch analysis
    - Configurable sentiment scales and output formats
    """

    def __init__(
        self,
        cache_dir: str = ".cache/sentiment",
        model: str = DEFAULT_LLM_MODEL,
        sentiment_scale: int = 10,
        use_cache: bool = True,
        api_key: Optional[str] = None,
    ):
        """
        Initialize sentiment analyzer.

        Args:
            cache_dir:        Directory for caching sentiment scores (daily).
            model:            Claude model to use.
            sentiment_scale:  Maximum sentiment score (1-N scale).
            use_cache:        Whether to use daily caching.
            api_key:          Anthropic API key (uses ANTHROPIC_API_KEY env var if None).
        """
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.sentiment_scale = sentiment_scale
        self.use_cache = use_cache
        self._rate_limiter = _RateLimiter()

        # Create cache directory with restricted permissions (owner only)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.cache_dir, stat.S_IRWXU)

        # Initialize Anthropic client and validate the key immediately
        self.client = anthropic.Anthropic(api_key=api_key)
        self._validate_api_key()

    def _validate_api_key(self) -> None:
        """Fail fast if the API key is missing or invalid."""
        try:
            self.client.models.list()
        except anthropic.AuthenticationError as e:
            raise ValueError(f"Invalid or missing Anthropic API key: {e}") from e
        except anthropic.APIConnectionError:
            # Network unavailable — do not block init, will fail at query time
            logger.warning("Could not reach Anthropic API during init (network issue). Continuing.")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache_key(self, assets: List[str], date: Optional[str] = None) -> str:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        key_str = f"{','.join(sorted(assets))}_{date}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            cached_date = data.get('date')
            today = datetime.now().strftime("%Y-%m-%d")
            if cached_date == today:
                return data
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error reading cache file {cache_file}: {e}")

        return None

    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        if not self.use_cache:
            return

        # Enforce cache size limit before writing
        try:
            total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
            new_bytes = len(json.dumps(data).encode())
            if total_bytes + new_bytes > MAX_CACHE_SIZE_MB * 1024 * 1024:
                logger.warning("Cache size limit reached; clearing entries older than 1 day.")
                self.clear_cache(days_old=1)
        except OSError as e:
            logger.warning(f"Could not check cache size: {e}")

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.chmod(cache_file, stat.S_IRUSR | stat.S_IWUSR)  # 600
        except OSError as e:
            logger.warning(f"Error writing cache file {cache_file}: {e}")

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze_sentiment(
        self,
        assets: List[str],
        date: Optional[str] = None,
        context: str = "",
    ) -> Dict[str, float]:
        """
        Analyze sentiment for given assets on a specific date.

        Args:
            assets:  List of asset tickers (e.g., ['AAPL', 'GOOGL', 'MSFT']).
            date:    Analysis date (YYYY-MM-DD, default: today).
            context: Additional context for the analysis prompt.

        Returns:
            Dictionary mapping assets to sentiment scores (0-1 normalised).
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        cache_key = self._get_cache_key(assets, date)
        cached = self._load_from_cache(cache_key)
        if cached:
            return {asset: cached['sentiments'].get(asset, 0.5) for asset in assets}

        prompt = self._build_prompt(assets, date, context)

        try:
            self._rate_limiter.wait()
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
            sentiments = self._parse_sentiment_response(response_text, assets)

            cache_data = {
                'date': date,
                'assets': assets,
                'sentiments': sentiments,
                'timestamp': datetime.now().isoformat(),
            }
            self._save_to_cache(cache_key, cache_data)
            return sentiments

        except anthropic.APIStatusError as e:
            logger.error(f"Anthropic API error (status {e.status_code}): {e.message}")
        except anthropic.APIConnectionError as e:
            logger.error(f"Anthropic connection error: {e}")

        return {asset: 0.5 for asset in assets}

    def _build_prompt(self, assets: List[str], date: str, context: str) -> str:
        assets_str = ", ".join(assets)
        prompt = f"""Analyze the market sentiment for the following assets on {date}:
Assets: {assets_str}

{"Additional context: " + context if context else ""}

For each asset, provide a sentiment score from 1 to {self.sentiment_scale}, where:
- 1 = Extremely Bearish
- {self.sentiment_scale // 2} = Neutral
- {self.sentiment_scale} = Extremely Bullish

Consider recent news, market trends, technical indicators, and macroeconomic factors.

Format your response as a JSON object like this:
{{
  "ASSET1": 7,
  "ASSET2": 4,
  ...
}}

Respond with ONLY the JSON object, no other text."""
        return prompt

    def _parse_sentiment_response(self, response: str, assets: List[str]) -> Dict[str, float]:
        """Parse Claude's response, validate structure, and normalise scores to 0-1."""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in LLM response")

            json_str = response[json_start:json_end]
            scores = json.loads(json_str)

            if not isinstance(scores, dict):
                raise ValueError(f"Expected a JSON object, got {type(scores).__name__}")

            # Validate all values are numeric and in expected range
            for key, val in scores.items():
                if not isinstance(val, (int, float)):
                    raise ValueError(f"Score for '{key}' is not numeric: {val!r}")
                if not (1 <= val <= self.sentiment_scale):
                    raise ValueError(
                        f"Score for '{key}' is out of range [1, {self.sentiment_scale}]: {val}"
                    )

            normalised: Dict[str, float] = {}
            for asset in assets:
                if asset in scores:
                    normalised[asset] = (scores[asset] - 1) / (self.sentiment_scale - 1)
                else:
                    logger.warning(f"Asset '{asset}' missing from LLM response; defaulting to neutral.")
                    normalised[asset] = 0.5

            return normalised

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing sentiment response: {e}")
            return {asset: 0.5 for asset in assets}

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def batch_analyze(
        self,
        assets: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze sentiment for a date range.

        Args:
            assets:     List of asset tickers.
            start_date: Start date (YYYY-MM-DD).
            end_date:   End date (YYYY-MM-DD).

        Returns:
            Dictionary mapping dates to {asset: score} dictionaries.
        """
        results: Dict[str, Dict[str, float]] = {}

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            results[date_str] = self.analyze_sentiment(assets, date_str)
            current += timedelta(days=1)

        return results

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self, days_old: int = 30) -> None:
        """Remove cache files older than `days_old` days."""
        if not self.use_cache:
            return

        cutoff = datetime.now() - timedelta(days=days_old)

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if mtime < cutoff:
                    cache_file.unlink()
            except OSError as e:
                logger.warning(f"Error deleting cache file {cache_file}: {e}")


__all__ = ['SentimentAnalyzer']
