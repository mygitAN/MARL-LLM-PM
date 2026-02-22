#!/usr/bin/env python3
"""
CLI entry point for MARL-LLM-PM portfolio management system.

Examples:
    python main.py backtest --config configs/default.yaml --assets AAPL GOOGL MSFT
    python main.py test --config configs/default.yaml
    python main.py train --episodes 100 --assets AAPL GOOGL --start-date 2023-01-01
"""

import argparse
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from marl_llm_pm import (
    PortfolioEnv,
    DummyAgent,
    AgentCoordinator,
    SentimentAnalyzer,
    Backtester,
    ConfigManager,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent
_TICKER_RE = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')


def validate_config_path(config_arg: str) -> Path:
    """Resolve config path and ensure it stays within the project and is a YAML file."""
    path = Path(config_arg).resolve()
    if path.suffix.lower() not in {'.yaml', '.yml'}:
        raise ValueError(f"Config file must be a .yaml/.yml file, got: {path.suffix}")
    if not str(path).startswith(str(_PROJECT_ROOT)):
        raise ValueError(f"Config path must be inside the project directory: {path}")
    return path


def validate_ticker(ticker: str) -> str:
    """Validate a single ticker symbol."""
    if not _TICKER_RE.match(ticker):
        raise ValueError(f"Invalid ticker symbol '{ticker}'. Expected 1-5 uppercase letters.")
    return ticker


def validate_date(date_str: str) -> str:
    """Validate a date string in YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date '{date_str}'. Expected format: YYYY-MM-DD")
    return date_str


def load_market_data(
    assets: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download market data using yfinance."""
    logger.info(f"Downloading data for {len(assets)} asset(s)...")

    data = yf.download(
        assets,
        start=start_date,
        end=end_date,
        progress=False,
    )
    
    # Handle single asset case
    if len(assets) == 1:
        data = data[['Close']].rename(columns={'Close': assets[0]})
    else:
        data = data['Close']
    
    logger.info(f"Loaded {len(data)} days of data")
    return data


def cmd_backtest(args, config: ConfigManager) -> None:
    """Run backtest with configured parameters."""
    logger.info("Starting backtest...")
    
    # Get assets and dates
    raw_assets = args.assets or config.get('environment', 'assets')
    try:
        assets = [validate_ticker(t) for t in raw_assets]
    except ValueError as e:
        logger.error(f"Invalid ticker: {e}")
        return

    raw_start = args.start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    raw_end = args.end_date or datetime.now().strftime('%Y-%m-%d')
    try:
        start_date = validate_date(raw_start)
        end_date = validate_date(raw_end)
    except ValueError as e:
        logger.error(f"Invalid date: {e}")
        return

    # Load market data
    try:
        price_data = load_market_data(assets, start_date, end_date)
    except (IOError, KeyError, ValueError) as e:
        logger.error(f"Failed to load market data: {e}")
        return
    
    # Create environment
    env = PortfolioEnv(
        asset_names=assets,
        initial_portfolio_value=config.get('backtesting', 'initial_portfolio_value'),
        max_steps=len(price_data),
        transaction_cost=config.get('backtesting', 'transaction_cost'),
    )
    env.set_market_data(price_data)
    
    # Create agents
    agents = [
        DummyAgent(f"agent_{i}", n_assets=len(assets))
        for i in range(config.get('agents', 'n_agents'))
    ]
    coordinator = AgentCoordinator(
        agents,
        aggregation_method=config.get('agents', 'aggregation_method')
    )
    
    # Weight calculator function
    def get_weights(observation):
        return coordinator.get_actions(observation)
    
    # Run backtest
    backtester = Backtester(
        initial_capital=config.get('backtesting', 'initial_capital'),
        transaction_cost=config.get('backtesting', 'transaction_cost'),
    )
    
    try:
        results = backtester.run(price_data, get_weights)
        logger.info("\n" + results.summary())

        # Save results
        results_path = Path(config.get('logging', 'results_dir', './results'))
        results_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_path / f"backtest_{timestamp}.csv"

        results_df = pd.DataFrame({
            'portfolio_value': results.portfolio_values,
            'returns': np.concatenate([[0], results.returns]),
        })
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")

    except (ValueError, RuntimeError) as e:
        logger.error(f"Backtest failed: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error during backtest: {e}", exc_info=True)
        raise


def cmd_train(args, config: ConfigManager) -> None:
    """Train agents on historical data."""
    logger.info("Training not yet implemented")
    logger.info(
        "To implement training, extend agents to inherit from BaseAgent "
        "and implement learning logic in the update() method"
    )


def cmd_analyze(args, config: ConfigManager) -> None:
    """Analyze sentiment for given assets."""
    if not config.get('llm', 'enabled'):
        logger.error("LLM is disabled in configuration")
        return
    
    raw_assets = args.assets or config.get('environment', 'assets')
    try:
        assets = [validate_ticker(t) for t in raw_assets]
    except ValueError as e:
        logger.error(f"Invalid ticker: {e}")
        return

    raw_date = args.date or datetime.now().strftime('%Y-%m-%d')
    try:
        date = validate_date(raw_date)
    except ValueError as e:
        logger.error(f"Invalid date: {e}")
        return

    logger.info(f"Analyzing sentiment for {len(assets)} asset(s) on {date}...")

    analyzer = SentimentAnalyzer(
        cache_dir=config.get('llm', 'cache_dir'),
        model=config.get('llm', 'model'),
        use_cache=config.get('llm', 'cache_enabled'),
    )

    try:
        sentiments = analyzer.analyze_sentiment(assets, date)

        # Display results
        logger.info("Sentiment Analysis Results:")
        logger.info("-" * 40)
        for asset, score in sentiments.items():
            sentiment_text = "Bullish" if score > 0.5 else "Bearish" if score < 0.5 else "Neutral"
            logger.info(f"{asset:8s}: {score:.2f} ({sentiment_text})")

    except (ValueError, RuntimeError) as e:
        logger.error(f"Sentiment analysis failed: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error during sentiment analysis: {e}", exc_info=True)
        raise


def cmd_test(args, config: ConfigManager) -> None:
    """Run pytest test suite."""
    import subprocess
    
    logger.info("Running test suite...")
    
    test_dir = Path(__file__).parent / 'tests'
    cmd = ['python', '-m', 'pytest', str(test_dir), '-v']
    
    if args.coverage:
        cmd.append('--cov=src/marl_llm_pm')
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, shell=False)
        sys.exit(result.returncode)
    except FileNotFoundError as e:
        logger.error(f"Test runner not found (is pytest installed?): {e}")
    except OSError as e:
        logger.error(f"Test execution failed: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-Agent Reinforcement Learning with LLM for Portfolio Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  python main.py backtest --assets AAPL GOOGL MSFT
  python main.py analyze --assets AAPL GOOGL --date 2024-01-15
  python main.py test --coverage
  python main.py train --episodes 100
        '''
    )
    
    parser.add_argument(
        '--config',
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--assets', nargs='+', help='Asset tickers')
    backtest_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train agents')
    train_parser.add_argument('--episodes', type=int, help='Number of episodes')
    train_parser.add_argument('--assets', nargs='+', help='Asset tickers')
    train_parser.set_defaults(func=cmd_train)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze sentiment')
    analyze_parser.add_argument('--assets', nargs='+', help='Asset tickers')
    analyze_parser.add_argument('--date', help='Analysis date (YYYY-MM-DD)')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--coverage', action='store_true', help='Run with coverage')
    test_parser.set_defaults(func=cmd_test)
    
    args = parser.parse_args()
    
    # Validate and load configuration
    try:
        config_path = validate_config_path(args.config)
    except ValueError as e:
        parser.error(str(e))
    config = ConfigManager(str(config_path))
    
    # Execute command
    if args.command is None:
        parser.print_help()
    else:
        args.func(args, config)


if __name__ == '__main__':
    main()
