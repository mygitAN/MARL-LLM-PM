import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SplitWindow:
    """Walk-forward split windows."""
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
    """Generate walk-forward train/val/test/holdout splits.
    
    Returns:
        train_df, val_df, test_windows (list), holdout_df
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    df = df.sort_index()

    # Extract holdout window (last N months)
    holdout_end = df.index.max()
    holdout_start = holdout_end - pd.DateOffset(months=holdout_months)
    dev_df = df[df.index < holdout_start]
    holdout_df = df[df.index >= holdout_start]

    # Split dev into train/val/test
    n = len(dev_df)
    n_train = int(n * split_train)
    n_val = int(n * split_val)

    train_df = dev_df.iloc[:n_train]
    val_df = dev_df.iloc[n_train : n_train + n_val]
    test_df = dev_df.iloc[n_train + n_val :]

    # Non-overlapping test windows
    windows = []
    start = test_df.index.min()
    end = test_df.index.max()
    cur = start
    while cur <= end:
        nxt = cur + pd.DateOffset(months=test_interval_months)
        win = test_df[(test_df.index >= cur) & (test_df.index < nxt)]
        if len(win) > 0:
            windows.append(win)
        cur = nxt

    return train_df, val_df, windows, holdout_df
