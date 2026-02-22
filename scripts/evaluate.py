"""Evaluate a trained MARL-LLM-PM model on held-out data."""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def validate_config_path(config_arg: str) -> Path:
    path = Path(config_arg).resolve()
    if path.suffix.lower() not in {'.yaml', '.yml'}:
        raise ValueError(f"Config must be a .yaml/.yml file, got: {path.suffix}")
    if not str(path).startswith(str(_PROJECT_ROOT)):
        raise ValueError(f"Config path must be inside the project directory: {path}")
    return path


def validate_date(date_str: str) -> str:
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date '{date_str}'. Expected format: YYYY-MM-DD")
    return date_str


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained MARL-LLM-PM agents")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--start", type=str, required=True, help="Evaluation start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="Evaluation end date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="outputs/evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        validate_config_path(args.config)
        validate_date(args.start)
        validate_date(args.end)
    except ValueError as e:
        logger.error(f"Invalid argument: {e}")
        raise SystemExit(1)

    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"Period: {args.start} to {args.end}")
    # TODO: load model, run backtest, compute metrics, save report


if __name__ == "__main__":
    main()
