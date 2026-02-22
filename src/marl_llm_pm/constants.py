"""Central constants to avoid duplication across the codebase."""

# LLM
DEFAULT_LLM_MODEL = "claude-3-5-sonnet-20241022"

# Cache
MAX_CACHE_SIZE_MB = 100

# API rate limiting
API_RATE_LIMIT_PER_MINUTE = 10

# Numerical stability
EPSILON = 1e-8

__all__ = [
    "DEFAULT_LLM_MODEL",
    "MAX_CACHE_SIZE_MB",
    "API_RATE_LIMIT_PER_MINUTE",
    "EPSILON",
]
