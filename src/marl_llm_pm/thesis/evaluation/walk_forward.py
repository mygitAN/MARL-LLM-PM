"""Walk-forward evaluation schedule.

Implements the thesis protocol:
  - Proportional train / val / test split on the development set
  - Non-overlapping rolling test windows of fixed duration
  - Sealed holdout (last N months, never touched during development)
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SplitWindow:
    """A single walk-forward fold."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    holdout: pd.DataFrame


def proportional_walk_forward(
    df: pd.DataFrame,
    split_train: float = 0.60,
    split_val: float = 0.20,
    split_test: float = 0.20,
    test_interval_months: int = 6,
    holdout_months: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame], pd.DataFrame]:
    """
    Build walk-forward evaluation splits aligned to the thesis protocol.

    Timeline
    --------
    |<-------- development set ------->|<-- holdout (sealed) -->|
    |< train (60%) >|< val (20%) >|< test (20%) >|

    The test portion is further divided into non-overlapping windows of
    `test_interval_months` length for rolling out-of-sample evaluation.

    Args:
        df:                   Full dataset with DatetimeIndex, sorted ascending.
        split_train:          Fraction of development data for training.
        split_val:            Fraction of development data for validation.
        split_test:           Fraction of development data for test (informational;
                              remainder after train+val is used).
        test_interval_months: Length of each rolling test window (months).
        holdout_months:       Length of the sealed holdout period at the end (months).

    Returns:
        (train_df, val_df, test_windows, holdout_df)
          train_df:     Training data.
          val_df:       Validation data.
          test_windows: List of non-overlapping test DataFrames.
          holdout_df:   Sealed holdout (do not touch until final evaluation).

    Raises:
        ValueError: If df does not have a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex.")

    if not (0 < split_train < 1 and 0 < split_val < 1 and split_train + split_val < 1):
        raise ValueError(
            f"split_train ({split_train}) + split_val ({split_val}) must be < 1 "
            f"and each must be in (0, 1)."
        )

    df = df.sort_index()

    # Seal holdout at the tail
    holdout_end = df.index.max()
    holdout_start = holdout_end - pd.DateOffset(months=holdout_months)
    dev_df = df[df.index < holdout_start]
    holdout_df = df[df.index >= holdout_start]

    if len(dev_df) == 0:
        raise ValueError(
            f"Development set is empty after removing {holdout_months}-month holdout. "
            "Reduce holdout_months or provide more data."
        )

    n = len(dev_df)
    n_train = int(n * split_train)
    n_val = int(n * split_val)

    train_df = dev_df.iloc[:n_train]
    val_df = dev_df.iloc[n_train: n_train + n_val]
    test_df = dev_df.iloc[n_train + n_val:]

    # Non-overlapping rolling test windows
    test_windows: List[pd.DataFrame] = []
    if len(test_df) > 0:
        cur = test_df.index.min()
        end = test_df.index.max()

        while cur <= end:
            nxt = cur + pd.DateOffset(months=test_interval_months)
            window = test_df[(test_df.index >= cur) & (test_df.index < nxt)]
            if len(window) > 0:
                test_windows.append(window)
            cur = nxt

    return train_df, val_df, test_windows, holdout_df


__all__ = ["proportional_walk_forward", "SplitWindow"]
