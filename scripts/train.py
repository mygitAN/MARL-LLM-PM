"""Entry point for training MARL agents."""

import argparse
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train MARL-LLM-PM agents")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/runs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Training with config: {config_path}")
    print(f"Output dir: {args.output_dir}")
    # TODO: initialise environment, agents, and training loop


if __name__ == "__main__":
    main()
