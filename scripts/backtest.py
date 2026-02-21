"""Run a full backtest using trained agents and historical data."""

import argparse


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
    print(f"Running backtest: {args.start} → {args.end}")
    print(f"Tickers: {args.tickers}")
    # TODO: load data, initialise agents, run episode loop, save results


if __name__ == "__main__":
    main()
