from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SleeveReturnsBuildSpec:
    """Configuration for building sleeve returns from index levels."""

    date_col: str = "date"
    method: str = "simple"  # "simple" or "log"
    dropna: str = "any"  # "any" or "all" (row-wise)
    min_history_rows: int = 30


def load_index_levels_csv(
    csv_path: str | Path,
    *,
    date_col: str = "date",
    sleeves: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load index *levels* (prices) for each sleeve.

    Expected CSV format (wide):
        date, MOMENTUM, VALUE, QUALITY
        2010-01-04, 100.0, 100.0, 100.0
        ...

    - The date column can be named anything via `date_col`.
    - All other columns are treated as sleeve index levels.
    - Missing values are allowed; they will be handled in `levels_to_returns`.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Index-level CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(
            f"date_col='{date_col}' not found in columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        bad = df.loc[df[date_col].isna()].head(5)
        raise ValueError(
            "Failed to parse some dates. Here are a few offending rows:\n"
            f"{bad.to_string(index=False)}"
        )

    df = df.set_index(date_col).sort_index()

    if sleeves is not None:
        sleeves = list(sleeves)
        missing = [c for c in sleeves if c not in df.columns]
        if missing:
            raise ValueError(f"Requested sleeves missing in CSV: {missing}")
        df = df[sleeves]

    # Coerce numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def levels_to_returns(
    levels: pd.DataFrame,
    *,
    method: str = "simple",
    dropna: str = "any",
    min_history_rows: int = 30,
) -> pd.DataFrame:
    """Convert index levels to sleeve returns."""
    if method not in {"simple", "log"}:
        raise ValueError("method must be 'simple' or 'log'")
    if dropna not in {"any", "all"}:
        raise ValueError("dropna must be 'any' or 'all'")

    levels = levels.copy()

    if method == "log":
        if (levels <= 0).any().any():
            bad_cols = [c for c in levels.columns if (levels[c] <= 0).any()]
            raise ValueError(
                "Log-returns require strictly positive index levels. "
                f"Found non-positive values in: {bad_cols}"
            )
        rets = np.log(levels).diff()
    else:
        rets = levels.pct_change()

    rets = rets.iloc[1:]

    if dropna == "any":
        rets = rets.dropna(axis=0, how="any")
    else:
        rets = rets.dropna(axis=0, how="all")

    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how=dropna)

    if len(rets) < min_history_rows:
        raise ValueError(
            f"Not enough rows after cleaning to build returns: {len(rets)}. "
            f"Increase history or relax cleaning. (min_history_rows={min_history_rows})"
        )

    return rets


def build_sleeve_returns_csv(
    *,
    index_levels_csv: str | Path,
    out_csv: str | Path,
    sleeves: Optional[Iterable[str]] = None,
    spec: SleeveReturnsBuildSpec = SleeveReturnsBuildSpec(),
) -> Path:
    """One-shot helper to create `sleeve_returns.csv` from index levels."""
    levels = load_index_levels_csv(
        index_levels_csv, date_col=spec.date_col, sleeves=sleeves
    )
    rets = levels_to_returns(
        levels,
        method=spec.method,
        dropna=spec.dropna,
        min_history_rows=spec.min_history_rows,
    )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rets.to_csv(out_csv, index_label="date")
    return out_csv
