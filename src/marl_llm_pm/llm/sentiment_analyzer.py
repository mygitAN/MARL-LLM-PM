"""Claude-powered sentiment analysis with daily caching."""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import anthropic


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
        model: str = "claude-3-5-sonnet-20241022",
        sentiment_scale: int = 10,  # 1-10 sentiment scale
        use_cache: bool = True,
        api_key: Optional[str] = None,
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            cache_dir: Directory for caching sentiment scores (daily)
            model: Claude model to use
            sentiment_scale: Maximum sentiment score (1-N scale)
            use_cache: Whether to use daily caching
            api_key: Anthropic API key (uses env var if not provided)
        """
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.sentiment_scale = sentiment_scale
        self.use_cache = use_cache
        
        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def _get_cache_key(self, assets: List[str], date: Optional[str] = None) -> str:
        """Generate cache key for sentiment query."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        key_str = f"{','.join(sorted(assets))}_{date}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load sentiment scores from cache if available."""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is from today
                    cached_date = data.get('date')
                    today = datetime.now().strftime("%Y-%m-%d")
                    if cached_date == today:
                        return data
            except Exception as e:
                print(f"Error reading cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """Save sentiment scores to cache."""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error writing cache: {e}")
    
    def analyze_sentiment(
        self,
        assets: List[str],
        date: Optional[str] = None,
        context: str = "",
    ) -> Dict[str, float]:
        """
        Analyze sentiment for given assets on a specific date.
        
        Args:
            assets: List of asset tickers (e.g., ['AAPL', 'GOOGL', 'MSFT'])
            date: Analysis date (YYYY-MM-DD format, default: today)
            context: Additional context for sentiment analysis
            
        Returns:
            Dictionary mapping assets to sentiment scores (0-1 normalized)
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Check cache
        cache_key = self._get_cache_key(assets, date)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            return {
                asset: cached_result['sentiments'].get(asset, 0.5)
                for asset in assets
            }
        
        # Query Claude
        prompt = self._build_prompt(assets, date, context)
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse response
            response_text = message.content[0].text
            sentiments = self._parse_sentiment_response(response_text, assets)
            
            # Cache result
            cache_data = {
                'date': date,
                'assets': assets,
                'sentiments': sentiments,
                'raw_response': response_text,
                'timestamp': datetime.now().isoformat(),
            }
            self._save_to_cache(cache_key, cache_data)
            
            return sentiments
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            # Return neutral sentiment on error
            return {asset: 0.5 for asset in assets}
    
    def _build_prompt(self, assets: List[str], date: str, context: str) -> str:
        """Build prompt for Claude sentiment analysis."""
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
        """Parse Claude's sentiment response and normalize to 0-1."""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            scores = json.loads(json_str)
            
            # Normalize to 0-1 and ensure all assets have scores
            normalized = {}
            for asset in assets:
                if asset in scores:
                    score = scores[asset]
                    # Normalize from 1-N scale to 0-1
                    normalized[asset] = (score - 1) / (self.sentiment_scale - 1)
                else:
                    # Default to neutral if missing
                    normalized[asset] = 0.5
            
            return normalized
            
        except Exception as e:
            print(f"Error parsing sentiment response: {e}")
            return {asset: 0.5 for asset in assets}
    
    def batch_analyze(
        self,
        assets: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze sentiment for multiple dates.
        
        Args:
            assets: List of asset tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping dates to asset sentiment dictionaries
        """
        results = {}
        
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            results[date_str] = self.analyze_sentiment(assets, date_str)
            current += timedelta(days=1)
        
        return results
    
    def clear_cache(self, days_old: int = 30) -> None:
        """Clear cache files older than specified days."""
        if not self.use_cache:
            return
        
        cutoff = datetime.now() - timedelta(days=days_old)
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if mtime < cutoff:
                    cache_file.unlink()
            except Exception as e:
                print(f"Error deleting cache file: {e}")


__all__ = ['SentimentAnalyzer']
