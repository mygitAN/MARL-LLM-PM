"""Evaluate a trained MARL-LLM-PM model on held-out data."""

import argparse
from pathlib import Path


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
    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Period: {args.start} → {args.end}")
    # TODO: load model, run backtest, compute metrics, save report


if __name__ == "__main__":
    main()
