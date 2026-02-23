"""Convert Bloomberg factor-index level CSVs into sleeve return streams."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SleeveReturnsBuildSpec:
    """Parameters controlling how index levels are converted to returns."""

    date_col: str = "date"
    method: str = "simple"       # "simple" (pct_change) or "log" (log-diff)
    dropna: str = "any"          # "any" | "all" — passed to DataFrame.dropna(how=)
    min_history_rows: int = 30   # raise if output has fewer rows than this


def load_index_levels_csv(
    csv_path,
    *,
    date_col: str = "date",
    sleeves: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load Bloomberg index-level CSV and return a DatetimeIndex DataFrame.

    Args:
        csv_path: Path to the CSV containing index levels (one column per sleeve).
        date_col: Name of the date column; may also be the index.
        sleeves: Optional list of column names to retain; if None, retain all.

    Returns:
        DataFrame with DatetimeIndex and float64 columns.
    """
    path = Path(csv_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Index levels CSV not found: {path}")
    if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        raise ValueError(f"Expected a CSV file, got: {path.suffix}")

    df = pd.read_csv(path)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    if sleeves is not None:
        missing = [s for s in sleeves if s not in df.columns]
        if missing:
            raise ValueError(f"Sleeve columns not found in CSV: {missing}")
        df = df[sleeves]

    df = df.apply(pd.to_numeric, errors="coerce")
    logger.info(f"Loaded index levels: {df.shape} from {path.name}")
    return df


def levels_to_returns(
    levels: pd.DataFrame,
    *,
    method: str = "simple",
    dropna: str = "any",
    min_history_rows: int = 30,
) -> pd.DataFrame:
    """Convert price/index levels to period returns.

    Args:
        levels: DataFrame of index levels (DatetimeIndex, one column per sleeve).
        method: "simple" for arithmetic pct_change; "log" for log-difference.
        dropna: Passed to DataFrame.dropna(how=) — "any" or "all".
        min_history_rows: Minimum number of rows required after dropna.

    Returns:
        DataFrame of period returns with the same columns and a shorter DatetimeIndex.
    """
    if method == "simple":
        returns = levels.pct_change()
    elif method == "log":
        returns = np.log(levels).diff()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'simple' or 'log'.")

    returns = returns.dropna(how=dropna)

    if len(returns) < min_history_rows:
        raise ValueError(
            f"Only {len(returns)} rows remain after computing returns "
            f"(minimum required: {min_history_rows}). "
            "Check your input data or lower min_history_rows."
        )

    logger.info(f"Computed {method} returns: {returns.shape}")
    return returns


def build_sleeve_returns_csv(
    *,
    index_levels_csv,
    out_csv,
    sleeves: Optional[List[str]] = None,
    spec: SleeveReturnsBuildSpec = SleeveReturnsBuildSpec(),
) -> Path:
    """End-to-end pipeline: load Bloomberg levels CSV → save sleeve returns CSV.

    Args:
        index_levels_csv: Path to the input index-levels CSV (from Bloomberg).
        out_csv: Destination path for the output sleeve-returns CSV.
        sleeves: Column names to retain; None = retain all.
        spec: Conversion parameters.

    Returns:
        Resolved Path of the written CSV.
    """
    out_path = Path(out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    levels = load_index_levels_csv(
        index_levels_csv,
        date_col=spec.date_col,
        sleeves=sleeves,
    )
    returns = levels_to_returns(
        levels,
        method=spec.method,
        dropna=spec.dropna,
        min_history_rows=spec.min_history_rows,
    )

    returns.index.name = spec.date_col
    returns.to_csv(out_path)
    logger.info(f"Sleeve returns saved to {out_path} ({len(returns)} rows)")
    return out_path


__all__ = [
    "SleeveReturnsBuildSpec",
    "load_index_levels_csv",
    "levels_to_returns",
    "build_sleeve_returns_csv",
]
