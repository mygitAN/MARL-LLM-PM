"""Walk-forward validation to prevent look-ahead bias."""

import pandas as pd
from dataclasses import dataclass
from typing import Callable


@dataclass
class WalkForwardResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: dict


class WalkForwardValidator:
    """
    Expanding-window or rolling-window walk-forward validation.

    Example
    -------
    validator = WalkForwardValidator(train_years=3, test_months=6)
    results = validator.run(data, train_fn, eval_fn)
    """

    def __init__(
        self,
        train_years: int = 3,
        test_months: int = 6,
        expanding: bool = True,
    ):
        self.train_years = train_years
        self.test_months = test_months
        self.expanding = expanding

    def generate_splits(self, index: pd.DatetimeIndex) -> list[tuple]:
        """Yield (train_idx, test_idx) pairs."""
        splits = []
        train_size = pd.DateOffset(years=self.train_years)
        test_size = pd.DateOffset(months=self.test_months)

        start = index.min()
        train_end = start + train_size

        while train_end + test_size <= index.max():
            test_end = train_end + test_size
            train_mask = (index >= start) & (index < train_end)
            test_mask = (index >= train_end) & (index < test_end)
            splits.append((index[train_mask], index[test_mask]))

            if not self.expanding:
                start += test_size
            train_end += test_size

        return splits

    def run(
        self,
        data: pd.DataFrame,
        train_fn: Callable,
        eval_fn: Callable,
    ) -> list[WalkForwardResult]:
        """
        Run walk-forward validation.

        Args:
            data:     Full dataset with DatetimeIndex.
            train_fn: fn(train_data) -> model
            eval_fn:  fn(model, test_data) -> dict of metrics

        Returns:
            List of WalkForwardResult for each fold.
        """
        splits = self.generate_splits(data.index)
        results = []

        for train_idx, test_idx in splits:
            model = train_fn(data.loc[train_idx])
            metrics = eval_fn(model, data.loc[test_idx])
            results.append(
                WalkForwardResult(
                    train_start=str(train_idx[0].date()),
                    train_end=str(train_idx[-1].date()),
                    test_start=str(test_idx[0].date()),
                    test_end=str(test_idx[-1].date()),
                    metrics=metrics,
                )
            )

        return results
