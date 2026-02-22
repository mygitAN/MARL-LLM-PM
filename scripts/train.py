"""Entry point for training MARL agents."""

import argparse
import logging
import yaml
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train MARL-LLM-PM agents")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/runs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        config_path = validate_config_path(args.config)
    except ValueError as e:
        logger.error(f"Invalid config path: {e}")
        raise SystemExit(1)

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        logger.error(f"Failed to load config: {e}")
        raise SystemExit(1)

    logger.info(f"Training with config: {config_path}")
    logger.info(f"Output dir: {args.output_dir}")
    # TODO: initialise environment, agents, and training loop


if __name__ == "__main__":
    main()
