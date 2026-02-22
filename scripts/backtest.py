"""Run a full backtest using trained agents and historical data."""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TICKER_RE = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')


def validate_config_path(config_arg: str) -> Path:
    path = Path(config_arg).resolve()
    if path.suffix.lower() not in {'.yaml', '.yml'}:
        raise ValueError(f"Config must be a .yaml/.yml file, got: {path.suffix}")
    if not str(path).startswith(str(_PROJECT_ROOT)):
        raise ValueError(f"Config path must be inside the project directory: {path}")
    return path


def validate_ticker(ticker: str) -> str:
    if not _TICKER_RE.match(ticker):
        raise ValueError(f"Invalid ticker '{ticker}'. Expected 1-5 uppercase letters.")
    return ticker


def validate_date(date_str: str) -> str:
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date '{date_str}'. Expected format: YYYY-MM-DD")
    return date_str


def parse_args():
    parser = argparse.ArgumentParser(description="Run MARL-LLM-PM backtest")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="outputs/backtest")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        validate_config_path(args.config)
        tickers = [validate_ticker(t) for t in args.tickers]
        start = validate_date(args.start)
        end = validate_date(args.end)
    except ValueError as e:
        logger.error(f"Invalid argument: {e}")
        raise SystemExit(1)

    logger.info(f"Running backtest: {start} to {end}")
    logger.info(f"Assets: {len(tickers)} ticker(s)")
    # TODO: load data, initialise agents, run episode loop, save results


if __name__ == "__main__":
    main()
