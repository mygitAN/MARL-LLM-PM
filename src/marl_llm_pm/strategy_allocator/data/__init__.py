"""Data utilities for the strategy-sleeve allocator.

This module is intentionally *data-source agnostic*.
For the thesis, you can generate sleeve (strategy) returns from:

1) Provider factor/strategy indices (recommended starting point).
2) Custom factor construction from an equity universe.

The code here focuses on (1): turning a CSV of index *levels* into a CSV
of sleeve returns that the StrategyAllocationEnv can consume.
"""

from .sleeve_returns_builder import (
    SleeveReturnsBuildSpec,
    load_index_levels_csv,
    levels_to_returns,
    build_sleeve_returns_csv,
)
